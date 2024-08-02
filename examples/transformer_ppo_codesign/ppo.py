import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from . import helper
from .evaluate import evaluate
from .envs import make_vec_envs
import numpy as np
from copy import deepcopy
from collections import deque
from utils.algo_utils import Structure
from .transformer.buffer import Buffer
import csv
import gym
import evogym.envs
from evogym import get_full_connectivity
from modular_envs.wrappers.modular_wrapper_codesign import modular_env
from simple_gif_codesign import gen_gif
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# PPO class
class PPO:
    def __init__(self, robots=None, termination_condition=None,pop_size=None, agent=None,
    verbose=True, ppo_args=None, save_path=None, save_replay=1):
        self.args = ppo_args
        self.verbose = verbose
        self.agent = agent
        self.pop_size = pop_size
        self.robots = []
        self.robots_tuple = robots
        self.real_robot = None
        self.total_iter = 0
        self.save_replay = save_replay
        # For multi robots
        if self.args.MULTI:
            # N threads, N robots
            for i in range(self.pop_size):  
                self.robots.append(Structure(*robots[i], i))
        # For a single robot
        else:
            # one robot, multi threads
            for i in range(self.pop_size):  
                self.robots.append(Structure(*robots[0], i))

        self.termination_condition = termination_condition
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.lr, eps=self.args.EPS)
        self.total_episode = 0
        
        print("Num params: {}".format(helper.num_params(self.agent.ac)))

        # Logger
        self.save_path_ = save_path
        self.header = ("iters", "timesteps",'total_episode',"eval_score",'train_score')
        # Record every robot's fitness
        if self.args.MULTI:
            for i in range(self.pop_size):
                self.header = self.header + (f"Robot_{i}_score",)

        self.fitness_log = open(self.save_path_ + f"/fitness_log.csv", 'w')
        self.fitness_logger = csv.DictWriter(self.fitness_log, fieldnames=self.header)
        self.tb_logger = SummaryWriter(self.save_path_)
        self.save_path_controllers = os.path.join(self.save_path_, "controllers")
        self.save_path_structures = os.path.join(self.save_path_, "structures")
        self.save_path_models = os.path.join(self.save_path_, "models")
        self.save_path_latent = os.path.join(self.save_path_, "latents")
        try:
            os.makedirs(self.save_path_controllers)
            os.makedirs(self.save_path_structures)
            os.makedirs(self.save_path_models)
            os.makedirs(self.save_path_latent)
        except:
            pass

        # Seed
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        
        # Set GPU device
        if self.args.cuda and torch.cuda.is_available() and self.args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)

        # Set replay buffer
        self.buffer = self.reset_buffer()
        self.buffer.to(self.device)
        self.use_aug_buffer = False
        if self.use_aug_buffer:
            self.aug_buffer = self.reset_buffer()
            self.aug_buffer.to(self.device)

    def reset_buffer(self):
        train_env = gym.make(self.args.env_name, body=self.robots_tuple[0][0], connections=self.robots_tuple[0][1])
        train_env = modular_env(env=train_env, body=self.robots_tuple[0][0])
        sequence_size = train_env.voxel_num
        obs_sample = train_env.reset()
        train_env.close()
        return Buffer(obs_sample, act_shape=sequence_size, num_envs=self.pop_size, cfg=self.args)

    def train(self, load_iter=None):
        # Make envs, the real training envs
        self.envs = make_vec_envs(self.args.env_name, self.robots, self.args.seed, 
                                self.args.GAMMA, self.device, ob=True, ret=True)
        start = time.time()
        obs = self.envs.reset()
        num_updates = int(self.args.num_env_steps) // self.args.TIMESTEPS // self.pop_size
        episode_rewards = deque(maxlen=10)
        design_success = deque(maxlen=10)
        logged_robots = deque(maxlen=4)
        logged_extra_act = deque(maxlen=4)
        
        # load model
        if load_iter is not None:
            start_j = load_iter + 1
            temp_path_model = os.path.join(self.save_path_models, f'iter_{str(load_iter)}.pt')
            cp = torch.load(temp_path_model)
            # self.agent.load_state_dict(cp["agent"])
            self.agent = cp["agent"]
            # self.agent.update_probs(cp["granularity"])
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.lr, eps=self.args.EPS)
            self.optimizer.load_state_dict(cp["optimizer"])
            helper.get_vec_normalize(self.envs).ob_rms = cp["ob_rms"]
            print(f"================== Load model =================== \n{temp_path_model}")

        else:
            start_j = 0

        for j in range(start_j, num_updates*4, 1):
            num_episode_cur_update = 1e-6
            num_steps_cur_update = 0
            if self.args.use_linear_lr_decay:
                # decrease learning rate linearly
                helper.update_linear_schedule(self.optimizer, j, num_updates*4,
                     self.args.lr) 

            # print("Collect experience !!!!!")
            for step in range(self.args.TIMESTEPS):
                # Sample actions
                val, act, logp = self.agent.uni_act(obs)
                next_obs, reward, done, infos = self.envs.step(act)
                for info_i, info in enumerate(infos):
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])                        
                    if info['stage'] == 'design':
                        if info['design_success'] == 'Done':
                            self.real_robot = info['real_design'] 
                            logged_robots.append(info['real_design'])
                            logged_extra_act.append(info['extra_act'])
                            design_success.append(1.0)
                        elif not info['design_success']:
                            design_success.append(0.0)
                    elif info['stage'] == 'act' and info['act_step'] == 1:
                        self.agent.update_species(info_i, val[info_i].item(), use_predicted_value=True, design_failed=False) 
                    
                    if done[info_i]:
                        self.total_episode +=1
                        num_episode_cur_update += 1
                        if not info['design_success']:
                            self.agent.update_species(info_i, -10.0, use_predicted_value=True, design_failed=True) 
                            self.agent.update_species(info_i, -10.0, use_predicted_value=False, design_failed=True) 
                        else:
                            self.agent.update_species(info_i, info['episode']['r'], use_predicted_value=False, design_failed=False) 
                num_steps_cur_update += 1

                masks = torch.tensor(
                        [[0.0] if done_ else [1.0] for done_ in done],
                        dtype=torch.float32,
                        device=self.device,
                    )
                timeouts = torch.tensor(
                        [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos],
                        dtype=torch.float32,
                        device=self.device,
                    )

                self.buffer.insert(obs, act, logp, val, reward, masks, timeouts)
                
                if self.use_aug_buffer:
                    obs_aug = deepcopy(obs)
                    # obs_aug['cluster'][:] = torch.eye(self.agent.wh**2, dtype=obs_aug['cluster'].dtype, device=obs_aug['cluster'].device).flatten()
                    obs_aug['cluster'] = obs_aug['cluster_aug']
                    self.aug_buffer.insert(obs_aug, act, logp, val, reward, masks, timeouts)
                
                obs = next_obs

            next_val = self.agent.get_value(obs)
            self.buffer.compute_returns(next_val)
            
            if self.use_aug_buffer:
                obs_aug = deepcopy(obs)
                obs_aug['cluster'] = obs_aug['cluster_aug']
                self.aug_buffer.insert(obs_aug, act, logp, val, reward, masks, timeouts)
                
                self.aug_buffer.compute_returns(next_val)

            # print("Begin training!!!!!")
            self.total_iter = j
            self.train_on_batch()
        
            # Evaluation 
            if (self.args.eval_interval is not None and j % self.args.eval_interval == 0):
                self.agent.update_agent(iteration = j) 
                if self.agent.use_hyperbolic_design:
                    
                    self.agent.log_hyperbolic_solutions = self.agent.log_hyperbolic_solutions[
                        -self.agent.hyperbolic_model.hyperbolic_optimizer.population_size:]
                    self.agent.hyperbolic_model.plot(save_path=os.path.join(self.save_path_latent, f"latent_{j}.png"),
                                                     log_hyperbolic_solutions = self.agent.log_hyperbolic_solutions)
               
                total_num_steps = (j + 1 - start_j) * self.pop_size * self.args.TIMESTEPS
                end = time.time()
                print("Updates {}, num timesteps {}, FPS {}".format(j, total_num_steps,
                            int(total_num_steps / (end - start)),))

                # print("Begin evaluation!!!!!")
                obs_rmss = helper.get_vec_normalize(self.envs).ob_rms
                if self.args.MULTI:
                    population_score = []
                    res = {"iters": j,"timesteps": total_num_steps,'total_episode':self.total_episode,
                          'train_score':np.average(episode_rewards)}
                    for i in range(self.pop_size):
                        reward = evaluate(num_evals=self.args.num_evals, uni_agent=self.agent,ob_rms=obs_rmss, env_name=self.args.env_name,
                                            init_robots=[self.robots[i]]*2,seed=self.args.seed, device=self.device)
                        res[f"Robot_{i}_score"] = reward 
                        population_score.append(reward)
                    res['eval_score'] = np.average(population_score)
                    avg_reward = np.average(population_score)
                else:
                    avg_reward = evaluate(num_evals=self.args.num_evals, uni_agent=self.agent,ob_rms=obs_rmss, env_name=self.args.env_name,
                                            init_robots=[self.robots[0]]*2,seed=self.args.seed, device=self.device)
                    res = {"iters": j,"timesteps": total_num_steps,'total_episode':self.total_episode, "eval_score":avg_reward
                        ,'train_score':np.average(episode_rewards)}

                self.fitness_logger.writerow(res)
                self.fitness_log.flush()
                self.tb_logger.add_scalar("eval_score", avg_reward, j)
                if len(episode_rewards) > 0:
                    self.tb_logger.add_scalar("train_score", np.average(episode_rewards), j)
                if len(design_success) > 0:
                    self.tb_logger.add_scalar("design_success", np.average(design_success), j)
                self.tb_logger.add_scalar("steps_per_episode_avg", total_num_steps/(self.total_episode + 1e-6), j)
                self.tb_logger.add_scalar("steps_per_episode", num_steps_cur_update/num_episode_cur_update, j)
                self.tb_logger.add_scalar("act_dist_std", self.agent.ac.act_dist_std.exp().mean().item(), j)
                self.tb_logger.add_scalar("act_abs_mean", self.buffer.act.abs().mean().item(), j)
                self.tb_logger.add_scalar("act_abs_std", self.buffer.act.abs().std().item(), j)
                self.tb_logger.add_scalar("act_mean", self.buffer.act.mean().item(), j)
                self.tb_logger.add_scalar("act_std", self.buffer.act.std().item(), j)                    
                self.tb_logger.flush()
                
                if self.verbose:
                    print(f'Mean reward: {avg_reward}')
                    for robot_i in range(len(logged_robots)):
                        print(f'robot: {robot_i}')
                        robot = logged_robots[robot_i]
                        print(robot)
                        print(logged_extra_act[robot_i])
                        
            
            if j % self.args.save_interval == 0:
                # Save controller
                temp_path_controller = os.path.join(self.save_path_controllers, f'iter_{str(j)}' + ".pt")
                torch.save([self.agent, getattr(helper.get_vec_normalize(self.envs), 'ob_rms', None)], temp_path_controller)
                
                # Save structure
                if self.real_robot is not None:
                    np.savez(os.path.join(self.save_path_structures, f"iter_{str(j)}.npz"), 
                             self.real_robot, get_full_connectivity(self.real_robot))

                # Save model
                temp_path_model = os.path.join(self.save_path_models, f'iter_{str(j)}.pt')
                torch.save({
                    "agent": self.agent, 
                    "optimizer": self.optimizer.state_dict(),
                    "ob_rms": getattr(helper.get_vec_normalize(self.envs), 'ob_rms', None), 
                }, temp_path_model)           
                print(f"================== Save model =================== \n{temp_path_model}")

                if self.save_replay > 0:
                    gen_gif(self.args.env_name, [self.args.seed], [j], False, 
                            EXPERIMENT_PARENT_DIR=self.save_path_) 
                
            # return
            if not self.termination_condition == None:
                if self.termination_condition(j):
                    self.envs.close()
                    if self.verbose:
                        print(f'Met termination condition ({j})...terminating...\n')
    
    def train_on_batch(self):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        value_epoch = 0
        approx_kl_epoch = 0

        adv = self.buffer.ret - self.buffer.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        
        if self.use_aug_buffer:
            adv_aug = self.aug_buffer.ret - self.aug_buffer.val
            adv_aug = (adv_aug - adv_aug.mean()) / (adv_aug.std() + 1e-5)

        num_updates = 1
        continue_count = 0
        total_count = 0
        for ep in range(self.args.EPOCHS):
            if np.random.uniform() < 0.5 or not self.use_aug_buffer:
                batch_sampler = self.buffer.get_sampler(adv)
            else:
                batch_sampler = self.aug_buffer.get_sampler(adv_aug)
            for batch in batch_sampler:
                # Reshape to do in a single forward pass for all steps
                val, logp, ent = self.agent(batch["obs"], batch["act"])
                clip_ratio = self.args.CLIP_EPS
                ratio = torch.exp(logp - batch["logp_old"])

                surr1 = ratio * batch["adv"]
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)*batch["adv"]
                
                pi_loss = -torch.min(surr1, surr2).mean()
                approx_kl = (logp - batch["logp_old"]).abs().mean() 

                total_count += 1

                if self.args.USE_CLIP_VALUE_FUNC:
                    val_pred_clip = batch["val"] + (val - batch["val"]).clamp(
                        -clip_ratio, clip_ratio
                    )
                    val_loss = (val - batch["ret"]).pow(2)
                    val_loss_clip = (val_pred_clip - batch["ret"]).pow(2)
                    val_loss = 0.5 * torch.max(val_loss, val_loss_clip).mean()
                else:
                    val_loss = 0.5 * (batch["ret"] - val).pow(2).mean()

                self.optimizer.zero_grad()
                loss = val_loss * self.args.VALUE_COEF + pi_loss - ent * self.args.ENTROPY_COEF
                if loss.requires_grad: # for NGE
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    
                    self.optimizer.step()

                value_loss_epoch += val_loss.item()
                action_loss_epoch += pi_loss.item()
                dist_entropy_epoch += ent.item()
                value_epoch += val.mean().item()
                approx_kl_epoch += approx_kl.item()

                num_updates += 1

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        value_epoch /= num_updates
        approx_kl_epoch /= num_updates

        print(f"PI loss: {action_loss_epoch:.4f}, value loss: {value_loss_epoch:.4f}, entropy: {dist_entropy_epoch:.4f}, value: {value_epoch:.4f}")
        
        self.tb_logger.add_scalar("pi_loss", action_loss_epoch, self.total_iter)
        self.tb_logger.add_scalar("value_loss", value_loss_epoch, self.total_iter)
        self.tb_logger.add_scalar("entropy", dist_entropy_epoch, self.total_iter)
        self.tb_logger.add_scalar("value", value_epoch, self.total_iter)
        self.tb_logger.add_scalar("approx_kl", approx_kl_epoch, self.total_iter)
        self.tb_logger.add_scalar("skip_ratio", continue_count/total_count, self.total_iter)