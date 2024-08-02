from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from .transformermodel import TransformerModel
from .distributions import DiagGaussian, Categorical
from .nge_design import SpeciesBank
from evogym.utils import _recursive_search
from scipy import stats
from collections import deque
from evogym.utils import has_actuator, is_connected

SCALE = 1 
STDVAR = 1 

class TransformerPPOAC(nn.Module):

    def __init__(
        self,
        modular_state_dim,
        modular_action_dim,
        design_state_dim,
        design_action_dim, 
        extra_design_action_dim, 
        sequence_size,
        other_feature_size,
        ppo_args=None,
        trans_args=None,
        ac_type=None,
        device=None,
        wh=8, 
    ):
        super(TransformerPPOAC, self).__init__()
        self.sequence_size = sequence_size
        self.input_state = [None] * self.sequence_size
        self.other_feature_size = other_feature_size
        self.state_dim = modular_state_dim
        self.action_dim = modular_action_dim
        self.design_state_dim = design_state_dim
        self.design_action_dim = design_action_dim
        self.extra_design_action_dim = extra_design_action_dim
        self.ppo_args = ppo_args
        self.trans_args = trans_args
        self.ac_type=ac_type
        self.device = device
        self.wh = wh
        assert self.sequence_size == self.wh ** 2

        self.preprocessed_state_dim = self.state_dim # 128

        self.v_net = TransformerModel(
            feature_size=self.preprocessed_state_dim,
            output_size=1, # self.action_dim, 
            sequence_size=self.sequence_size,
            other_feature_size=self.other_feature_size,
            ninp=trans_args.attention_embedding_size,
            nhead=trans_args.attention_heads,
            nhid= trans_args.attention_hidden_size,
            nlayers=trans_args.attention_layers * 2,
            dropout=trans_args.dropout_rate,
            args=trans_args,
            use_transformer=self.ac_type,
            final_nonlinearity=False,
            nonlinearity='relu',
            zero_initialize_final_layer=False,
        )

        self.mu_net = TransformerModel(
            feature_size=self.preprocessed_state_dim,
            output_size=self.action_dim,
            sequence_size=self.sequence_size,
            other_feature_size=self.other_feature_size,
            ninp=trans_args.attention_embedding_size,
            nhead=trans_args.attention_heads,
            nhid= trans_args.attention_hidden_size,
            nlayers=trans_args.attention_layers * 2,
            dropout=trans_args.dropout_rate,
            args=trans_args,
            use_transformer=self.ac_type,
            final_nonlinearity=False, 
            nonlinearity='tanh',
            zero_initialize_final_layer=False,
        )

        self.design_v_net = TransformerModel(
            feature_size=self.design_state_dim,
            output_size=1, 
            sequence_size=self.sequence_size,
            other_feature_size=self.other_feature_size,
            ninp=trans_args.attention_embedding_size//1,  
            nhead=trans_args.attention_heads,
            nhid= trans_args.attention_hidden_size//1,
            nlayers=trans_args.attention_layers,
            dropout=trans_args.dropout_rate,
            args=trans_args,
            use_transformer=self.ac_type,
            final_nonlinearity=False, 
            nonlinearity='relu',
            zero_initialize_final_layer=False,
        )

        self.design_mu_net = TransformerModel(
            feature_size=self.design_state_dim,
            output_size=self.design_action_dim + self.extra_design_action_dim, 
            sequence_size=self.sequence_size,
            other_feature_size=self.other_feature_size,
            ninp=trans_args.attention_embedding_size//1,
            nhead=trans_args.attention_heads,
            nhid= trans_args.attention_hidden_size//1,
            nlayers=trans_args.attention_layers,
            dropout=trans_args.dropout_rate,
            args=trans_args,
            use_transformer=self.ac_type,
            final_nonlinearity=False, 
            nonlinearity='tanh',
            zero_initialize_final_layer=False,
            # is_actor = True
        )

        self.act_dist_std = nn.Parameter(torch.ones(1, 1) * np.log(self.ppo_args.ACTION_STD), 
                                         requires_grad=not self.ppo_args.ACTION_STD_FIXED)
        self.design_mu_direct = None
        self.extra_design_mu_direct = None
        
    def forward(self, state, act=None, return_attention=False, design=False):
        if design:
            modular_state, other_state,act_mask,obs_padding, design_state, design_cluster = (
                state["modular"],
                state["other"],
                state["act_mask"],
                state["obs_mask"],
                state["design"],
                state["cluster"],
            )

            batch_size = design_state.shape[0]
            act_mask = act_mask.bool()
            obs_padding = obs_padding.bool()

            inpt = design_state.reshape(batch_size, self.sequence_size, -1) # [bs, N, J]

            # normally divided
            inpt = inpt.reshape(batch_size, self.sequence_size, -1).permute(1,0,2) # [N,bs,J]
            cur_obs_padding = obs_padding.reshape(batch_size, self.sequence_size)

            num_limbs = self.sequence_size - torch.sum(cur_obs_padding.int(), dim=1, keepdim=True)

            # module_vals shape: (batch_size, num_modular, J)
            module_vals, v_attention_maps = self.design_v_net(
                inpt, other_state, cur_obs_padding, return_attention=return_attention
            )
            module_vals = module_vals.squeeze(-1)
            # val shape: (batch_size, 1)
            # Zero out mask values
            module_vals = module_vals * (1 - cur_obs_padding.int())
            val = torch.divide(torch.sum(module_vals, dim=1, keepdim=True), num_limbs)

            # mu shape: (batch_size, num_modular, J)
            mu, mu_attention_maps = self.design_mu_net(
                inpt, other_state, cur_obs_padding, return_attention=return_attention
            )
            mu = mu.squeeze(-1)

            mu = mu.reshape(batch_size, self.sequence_size, -1).squeeze(-1)
            extra_mu = mu[:,:,self.design_action_dim:]
            mu = mu[:,:,:self.design_action_dim]

            self.design_mu_direct = mu.detach().mean(dim=0, keepdim=True)
            self.extra_design_mu_direct = extra_mu.detach().mean(dim=0, keepdim=True)


            pi = Categorical(logits=mu, uniform_prob=0.)
            if self.extra_design_action_dim == 1:
                extra_mu = extra_mu[:, 0:2, 0]
                loc = torch.arange(0, self.wh, device=extra_mu.device).float()
                loc = (loc - (self.wh-1)//2)/SCALE
                loc_probs = torch.exp( - (extra_mu.unsqueeze(2) - loc.reshape(1,1,self.wh))**2 / (2*STDVAR**2) )
                loc_probs = loc_probs / loc_probs.sum(dim=-1, keepdim=True)
                extra_pi = Categorical(probs=loc_probs, uniform_prob=0.)
            else:
                extra_pi = Categorical(logits=extra_mu, uniform_prob=0.) 
        else:
            modular_state, other_state,act_mask,obs_padding, design_cluster = (
                state["modular"],
                state["other"],
                state["act_mask"],
                state["obs_mask"],
                state["cluster"],
            )

            batch_size = modular_state.shape[0]
            act_mask = act_mask.bool()
            obs_padding = obs_padding.bool()

            inpt = modular_state.reshape(batch_size, self.sequence_size, -1) # [bs, N, J]

            # inpt = inpt_preprocess_net(inpt)
            attn_aggregation = True 
            if attn_aggregation:
                inpt = inpt.reshape(batch_size, self.sequence_size, -1).permute(1,0,2) # [N,bs,J] 
                cur_obs_padding = obs_padding.reshape(batch_size, self.sequence_size)
            else:
                inpt = (design_cluster.reshape(-1, self.sequence_size, self.sequence_size) @ inpt).permute(1,0,2) # [N, bs, J]
                cur_obs_padding = torch.isclose((1-(design_cluster.reshape(-1, self.sequence_size, self.sequence_size) @ (1-obs_padding.float()).unsqueeze(-1)).squeeze(-1)), torch.tensor(1).float())

            # module_vals shape: (batch_size, num_modular, J)
            module_vals, v_attention_maps = self.v_net(
                inpt, other_state, cur_obs_padding, return_attention=return_attention,
                design_cluster = design_cluster.reshape(-1, self.sequence_size, self.sequence_size), 
                attn_aggregation = attn_aggregation,
            )
            module_vals = module_vals.squeeze(-1)
            # val shape: (batch_size, 1)
            # Zero out mask values
            module_vals = module_vals * (1 - cur_obs_padding.int())
            num_limbs = self.sequence_size - torch.sum(cur_obs_padding.int(), dim=1, keepdim=True)
            val = torch.divide(torch.sum(module_vals, dim=1, keepdim=True), num_limbs)

            # mu shape: (batch_size, num_modular, J)
            mu, mu_attention_maps = self.mu_net(
                inpt, other_state, cur_obs_padding, return_attention=return_attention,
                design_cluster = design_cluster.reshape(-1, self.sequence_size, self.sequence_size), 
                attn_aggregation = attn_aggregation,
            )
            mu = mu.squeeze(-1)

            mu = (torch.logical_not(torch.isclose(design_cluster.reshape(-1, self.sequence_size, self.sequence_size).permute(0,2,1), 
                                                  torch.tensor(0.0))).float() @ mu.unsqueeze(-1)).squeeze(-1)
            
            mu = torch.tanh(mu) 
            std = self.act_dist_std.expand_as(mu).exp()
            pi = DiagGaussian(mu, std)
            extra_pi = None


        # In case next step is training
        if act is not None:
            if design:
                type_act = act % self.design_action_dim
                extra_act = torch.div(act, self.design_action_dim, rounding_mode='floor')
                logp = pi.log_prob(type_act)
                entropy = pi.entropy()
                if self.extra_design_action_dim == 1:
                    extra_entropy = extra_pi.entropy().sum(dim=-1)
                    logp[:,1:] = 0
                    entropy[:, 1:] = 0
                    idx_loc = extra_act.argmax(dim=-1)
                    xy = torch.stack([torch.div(idx_loc, self.wh, rounding_mode='floor'), idx_loc % self.wh], dim=-1)
                    extra_logp = extra_pi.log_prob(xy).sum(dim=-1)
                    logp[act_mask] = 0.0
                    logp = logp.sum(-1, keepdim=True) + extra_logp.unsqueeze(-1)
                    entropy[act_mask] = 0.0
                    entropy = entropy.sum(-1, keepdim=True) + extra_entropy.unsqueeze(-1)
                else:
                    extra_entropy = extra_pi.entropy()
                    logp[torch.isclose(extra_act, torch.tensor(0.0))] = 0.0 
                    entropy[torch.isclose(extra_act, torch.tensor(0.0))] = 0.0 
                    extra_logp = extra_pi.log_prob(extra_act)
                    logp = logp + extra_logp
                    logp[act_mask] = 0.0
                    logp = logp.sum(-1, keepdim=True) 
                    entropy = entropy + extra_entropy
                    entropy[act_mask] = 0.0
                    entropy = entropy.sum(-1, keepdim=True)
            else: 
                logp = pi.log_prob(act)
                logp[act_mask] = 0.0
                logp = (design_cluster.reshape(-1, self.sequence_size, self.sequence_size) @ logp.unsqueeze(-1)).squeeze(-1).sum(-1, keepdim=True)
                # logp = logp.sum(-1, keepdim=True)  
                entropy = pi.entropy()
                entropy[act_mask] = 0.0
                entropy = entropy.sum(-1, keepdim=True)
            return val, pi, extra_pi, logp, entropy
        else:
            if return_attention:
                return val, pi, extra_pi, v_attention_maps, mu_attention_maps
            else:
                return val, pi, extra_pi, None, None
    

class Agent(nn.Module):
    
    def __init__(self, actor_critic, wh=8, threads_num=1, eval_num=2):
        super(Agent, self).__init__()
        self.ac = actor_critic
        self.design_action_dim = self.ac.design_action_dim
        self.extra_design_action_dim = self.ac.extra_design_action_dim
        self.wh = wh # width & height # 
        self.sequence_size = self.wh ** 2
        self.use_nge_design = False 
        self.use_hyperbolic_design =  True 
        self.use_fixed_design = False 
        self.use_naive_cem = False
        self.threads_num = threads_num
        self.eval_num = eval_num
        self.start_granularity = 6 
        self.end_granularity = 8
        self.max_iters = 3000
        self.granularity = self.start_granularity 
        self.rank_correlation = 0
        self.design_diversity = 0
        self.rank_pvalue = 0
        self.change_granularity = deque([False]*5, maxlen=5) 
        self.avg_correlation = 0
        if self.use_nge_design:
            self.species_bank = SpeciesBank(wh, self.ac.design_action_dim, self.ac.extra_design_action_dim,
                                            granularity=self.granularity)
            self.sampled_species = [self.species_bank.sample_a_species() for _ in range(threads_num)]
            self.best_species = self.species_bank.best_species(eval_num)
            self.specified_species = []
        elif self.use_hyperbolic_design: 
            data_shape = [5, self.wh, self.wh, 1]
            from hyperbolic.model import HyperbolicModel
            self.hyperbolic_model = HyperbolicModel(*data_shape, device=self.ac.device)

            self.hyperbolic_zs = [[*self.hyperbolic_model.ask(), 0.0, 0.0] for _ in range(threads_num)]
            self.hyperbolic_best_zs = [[*self.hyperbolic_model.ask_mean(), 0.0, 0.0] for _ in range(eval_num)]
            self.log_hyperbolic_solutions = []
        elif self.use_fixed_design:
            data_shape = [5, self.wh, self.wh, 1]
            from hyperbolic.model import FixedModel
            self.fixed_model = FixedModel(*data_shape, device=self.ac.device)
        elif self.use_naive_cem:
            data_shape = [5, self.wh, self.wh, 1]
            from hyperbolic.model import NaiveCEMModel 
            self.hyperbolic_model = NaiveCEMModel(*data_shape, device=self.ac.device)
            self.hyperbolic_zs = [[*self.hyperbolic_model.ask(), 0.0, 0.0] for _ in range(threads_num)]
            self.hyperbolic_best_zs = [[*self.hyperbolic_model.ask_mean(), 0.0, 0.0] for _ in range(eval_num)]

    def update_species(self, i, reward, use_predicted_value=False, design_failed=False):
        if self.use_nge_design:
            if use_predicted_value: # act_step == 1
                self.sampled_species[i]['PredVal'] = reward
            else: # done of an episode
                species = self.sampled_species[i]
                species['LastRwd'] = reward
                self.species_bank.update_a_species(species, design_failed=design_failed)
                self.sampled_species[i] = self.species_bank.sample_a_species()
        elif self.use_hyperbolic_design:
            if not design_failed:
                self.hyperbolic_zs[i][2] += reward
                self.hyperbolic_zs[i][3] += 1

                if self.hyperbolic_zs[i][3] >= 0: 
                    tell_reward = self.hyperbolic_zs[i][2] / self.hyperbolic_zs[i][3]
                    self.log_hyperbolic_solutions.append((self.hyperbolic_model.manifold.expmap0(torch.from_numpy(
                    self.hyperbolic_zs[i][0]).to(self.ac.device)).cpu().numpy(), tell_reward))
                    self.hyperbolic_model.tell((self.hyperbolic_zs[i][1], tell_reward)) 
                    self.hyperbolic_zs[i] = [*self.hyperbolic_model.ask(), 0.0, 0.0]
            else:
                print("design fail !!!!!!")
                tell_reward = - 100 
                self.log_hyperbolic_solutions.append((self.hyperbolic_model.manifold.expmap0(torch.from_numpy(
                    self.hyperbolic_zs[i][0]).to(self.ac.device)).cpu().numpy(), tell_reward))
                
                self.hyperbolic_zs[i] = [*self.hyperbolic_model.ask(), 0.0, 0.0]
            
            self.hyperbolic_best_zs = [[*self.hyperbolic_model.ask_mean(), 0.0, 0.0] for _ in range(self.eval_num)]
        elif self.use_fixed_design:
            pass
        elif self.use_naive_cem:
            if not design_failed:
                self.hyperbolic_zs[i][2] += reward
                self.hyperbolic_zs[i][3] += 1

                if self.hyperbolic_zs[i][3] >= 0: 
                    tell_reward = self.hyperbolic_zs[i][2] / self.hyperbolic_zs[i][3]
                    self.hyperbolic_model.tell((self.hyperbolic_zs[i][1], tell_reward)) 
                    self.hyperbolic_zs[i] = [*self.hyperbolic_model.ask(), 0.0, 0.0]
            else:
                print("design fail !!!!!!")
                self.hyperbolic_zs[i] = [*self.hyperbolic_model.ask(), 0.0, 0.0]
            
            self.hyperbolic_best_zs = [[*self.hyperbolic_model.ask_mean(), 0.0, 0.0] for _ in range(self.eval_num)]
            

    def update_agent(self, iteration = None):
        if self.use_nge_design:
            self.species_bank.natural_selection()
            self.best_species = self.species_bank.best_species(self.eval_num)
            self.species_bank.log_information()
        elif self.use_hyperbolic_design:
            self.hyperbolic_model.update(iteration)
        elif self.use_fixed_design:
            pass
        elif self.use_naive_cem:
            self.hyperbolic_model.update(iteration)

    def forward(self, obs, act):
        index = obs['stage']
        batch_size = index.shape[0]
        ac_index = np.argwhere(np.isclose(index.cpu().numpy(),1)) # only act
        de_index = np.argwhere(np.isclose(index.cpu().numpy(),0)) # only design

        val = torch.zeros(batch_size,1).to(self.ac.device)
        logp = torch.zeros(batch_size,1).to(self.ac.device)
        ent = torch.zeros(batch_size,1).to(self.ac.device)

        ### ac batch
        if ac_index.shape[0]>0:
            if isinstance(obs, dict):
                ac_obs_batch ={}
                for ot, ov in obs.items():
                    ac_obs_batch[ot] = ov.view(-1, *ov.size()[1:])[ac_index[:,0]]

            ac_act_batch = act[ac_index[:,0]]
            ac_val, _, _, ac_logp, ac_ent = self.ac(ac_obs_batch, ac_act_batch, design=False)
            
            val[ac_index[:,0]] = ac_val
            logp[ac_index[:,0]] = ac_logp
            ent[ac_index[:,0]] = ac_ent
        
        ### de batch
        if de_index.shape[0]>0 and (not self.use_nge_design) \
            and (not self.use_hyperbolic_design) and (not self.use_fixed_design) and (not self.use_naive_cem):
            if isinstance(obs, dict):
                de_obs_batch ={}
                for ot, ov in obs.items():
                    de_obs_batch[ot] = ov.view(-1, *ov.size()[1:])[de_index[:,0]]

            de_act_batch = act[de_index[:,0]]
            ac_val, _, _, ac_logp, ac_ent = self.ac(de_obs_batch, de_act_batch, design=True)
            
            val[de_index[:,0]] = ac_val
            logp[de_index[:,0]] = ac_logp
            ent[de_index[:,0]] = ac_ent

        ent = ent.mean()
        # ent = ent.sum() / (torch.count_nonzero(ent) + 1e-6)
        return val, logp, ent

    @torch.no_grad()
    def uni_act(self, obs, mean_action=False):
        index = obs['stage']
        batch_size = index.shape[0]
        # ac_idx = np.argwhere(index.cpu().numpy()>0) # only act
        ac_idx = np.argwhere(np.isclose(index.cpu().numpy(),1)) # only act
        de_idx = np.argwhere(np.isclose(index.cpu().numpy(),0)) # only design
        
        val = torch.zeros(batch_size,1).to(self.ac.device)
        logp = torch.zeros(batch_size,1).to(self.ac.device)
        act = torch.zeros(batch_size,self.ac.sequence_size).to(self.ac.device)

        ### ac batch
        if ac_idx.shape[0]>0:
            if isinstance(obs, dict):
                ac_obs_batch ={}
                for ot, ov in obs.items():
                    ac_obs_batch[ot] = ov[ac_idx[:,0]]

            ac_val, ac_act, ac_logp = self.act(ac_obs_batch,mean_action=mean_action, design=False)

            val[ac_idx[:,0]] = ac_val
            logp[ac_idx[:,0]] = ac_logp

            # act[ac_idx[:,0]] = ac_act
            for j in range(ac_idx.shape[0]):
                act[ac_idx[:,0][j]] = ac_act[j]
        
        ### de batch
        if de_idx.shape[0]>0:
            if isinstance(obs, dict):
                de_obs_batch ={}
                for ot, ov in obs.items():
                    de_obs_batch[ot] = ov[de_idx[:,0]]

            ac_val, ac_act, ac_logp = self.act(de_obs_batch,mean_action=mean_action, design=True, de_idx=de_idx[:,0])

            if (not self.use_nge_design) and (not self.use_hyperbolic_design) and (not self.use_fixed_design) and (not self.use_naive_cem):
                val[de_idx[:,0]] = ac_val
                logp[de_idx[:,0]] = ac_logp

            # act[de_idx[:,0]] = ac_act.float()
            for j in range(de_idx.shape[0]):
                act[de_idx[:,0][j]] = ac_act[j]

        return val, act, logp

    @torch.no_grad()
    def act(self, obs, mean_action=False, design=False, de_idx=None):
        act_mask = obs["act_mask"].bool()
        if design and self.use_nge_design:
            if len(self.specified_species) > 0:
                new_design = np.array([self.specified_species[i]['Design'].reshape(self.sequence_size) 
                                       for i in range(len(self.specified_species))])
                new_extra_design = np.array([self.specified_species[i]['ExtraD'].reshape(self.sequence_size) 
                                             for i in range(len(self.specified_species))])
            else:    
                if mean_action:
                    new_design = np.array([self.best_species[i]['Design'].reshape(self.sequence_size) for i in de_idx])
                    new_extra_design = np.array([self.best_species[i]['ExtraD'].reshape(self.sequence_size) for i in de_idx])
                else:
                    new_design = np.array([self.sampled_species[i]['Design'].reshape(self.sequence_size) for i in de_idx])
                    new_extra_design = np.array([self.sampled_species[i]['ExtraD'].reshape(self.sequence_size) for i in de_idx])
            act = torch.from_numpy(new_design).to(act_mask.device)
            extra_act = torch.from_numpy(new_extra_design).to(act.device)   

            act = extra_act * self.design_action_dim + act
            logp = None
            val = None
        
        elif design and self.use_hyperbolic_design:
            if mean_action:
                zs = self.hyperbolic_best_zs
            else:
                zs = self.hyperbolic_zs
            
            for trial in range(100): # valid check
                act, extra_act = self.hyperbolic_model.generate_from_euclidean(
                    torch.from_numpy(np.stack([zs[i][0] for i in de_idx])).float().to(act_mask.device), 
                    mean_action=mean_action)
                test_act = act.reshape(act.shape[0], self.wh, self.wh)
                is_valid = [is_connected(test_act[i]) 
                            and has_actuator(test_act[i]) 
                            for i in range(test_act.shape[0])]
                if not np.all(is_valid):
                    for vi, va in enumerate(is_valid):
                        if not va:
                            zs[vi] = [*self.hyperbolic_model.ask(), 0.0, 0.0]
                else:
                    break
            if trial == 99:
                print("design will fail !!!")

            act = extra_act.reshape(len(de_idx), -1) * self.design_action_dim + act.reshape(len(de_idx), -1)   
            logp = None
            val = None
        
        elif design and self.use_fixed_design:
            act, extra_act = self.fixed_model.get_design()
            act = act.reshape(1,-1).repeat(len(de_idx), 1)
            extra_act = extra_act.reshape(1,-1).repeat(len(de_idx), 1)
            act = extra_act.reshape(len(de_idx), -1) * self.design_action_dim + act.reshape(len(de_idx), -1)   
            logp = None
            val = None
        
        elif design and self.use_naive_cem:
            if mean_action:
                zs = self.hyperbolic_best_zs
            else:
                zs = self.hyperbolic_zs
            
            for trial in range(5): # valid check
                act, extra_act = self.hyperbolic_model.generate_from_euclidean(
                    torch.from_numpy(np.stack([zs[i][0] for i in de_idx])).float().to(act_mask.device), 
                    mean_action=mean_action)
                test_act = act.reshape(act.shape[0], self.wh, self.wh)
                is_valid = [is_connected(test_act[i]) 
                            and has_actuator(test_act[i]) 
                            for i in range(test_act.shape[0])]
                if not np.all(is_valid):
                    for vi, va in enumerate(is_valid):
                        if not va:
                            zs[vi] = [*self.hyperbolic_model.ask(), 0.0, 0.0]
                else:
                    break
            if trial == 4:
                print("design will fail !!!")

            act = extra_act.reshape(len(de_idx), -1) * self.design_action_dim + act.reshape(len(de_idx), -1)   
            logp = None
            val = None

        elif design and (not self.use_nge_design) \
            and (not self.use_hyperbolic_design) and (not self.use_fixed_design) and (not self.use_naive_cem):
            val, pi, extra_pi, _, _ = self.ac(obs, design=design)
            if mean_action:
                act = pi.mean_sample()
                extra_act = extra_pi.mean_sample()
            else:
                act = pi.sample()
                extra_act = extra_pi.sample()
            if self.extra_design_action_dim == 1:
                extra_act = F.one_hot(extra_act[:, 0] * self.wh + extra_act[:, 1], num_classes=self.sequence_size)
                act[:] = act[:,0].unsqueeze(-1)

            act = extra_act * self.design_action_dim + act

            logp = pi.log_prob(act % self.design_action_dim) 
            if self.extra_design_action_dim == 1:
                logp[:,1:] = 0
                idx_loc = extra_act.argmax(dim=-1)
                xy = torch.stack([torch.div(idx_loc, self.wh, rounding_mode='floor'), idx_loc % self.wh], dim=-1)
                extra_logp = extra_pi.log_prob(xy).sum(dim=-1)
                logp[act_mask] = 0.0
                logp = logp.sum(-1, keepdim=True) + extra_logp.unsqueeze(-1)
            else:
                logp[torch.isclose(extra_act, torch.tensor(0))] = 0.0
                extra_logp = extra_pi.log_prob(extra_act)
                logp = logp + extra_logp
                logp[act_mask] = 0.0
                logp = logp.sum(-1, keepdim=True) 
            del pi
            del extra_pi

        else: # control
            val, pi, extra_pi, _, _ = self.ac(obs, design=design)
            if mean_action:
                act = pi.mean_sample()
            else:
                act = pi.sample()
            
            bs = act.shape[0]
            design_cluster = obs['cluster']
            select_1_list = []
            for i in range(bs):
                select_1 = act[i][design_cluster.reshape(-1, self.sequence_size, self.sequence_size)[i].max(dim=-1)[1]]
                select_1_list.append(select_1)
            select_1_all = torch.stack(select_1_list, dim=0)
            scatter_to_else = torch.logical_not(torch.isclose(design_cluster.reshape(-1, self.sequence_size, self.sequence_size).permute(0,2,1), torch.tensor(0).float())).float()
            act = (scatter_to_else @ select_1_all.unsqueeze(-1)).reshape(bs, -1)

            logp = pi.log_prob(act)
            logp[act_mask] = 0.0
            design_cluster = obs['cluster']
            logp = (design_cluster.reshape(-1, self.sequence_size, self.sequence_size) @ logp.unsqueeze(-1)).squeeze(-1).sum(-1, keepdim=True)
            # logp = logp.sum(-1, keepdim=True)  
            del pi

        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs):
        val, act, logp = self.uni_act(obs)
        return val