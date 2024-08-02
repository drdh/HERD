import numpy as np
from copy import deepcopy
import torch
from evogym.utils import is_connected, has_actuator, draw, get_uniform

MIN_LIFE_LENGTH = 1
MIN_MATURE_LENGTH = 1

class SpeciesBank:
    def __init__(self, wh, design_action_dim, extra_design_action_dim, granularity=1):
        self.current_generation = 1
        self.elimination_rate = 0.3
        self.reset_elimination_rate = 0.9
        self.maximum_num_species = 64
        self.start_num_species = 63
        self.num_total_species = 0
        self.granularity = granularity
        
        self.fixed_granularity = False

        self.wh = wh # width & height of robot
        self.design_action_dim = design_action_dim
        self.extra_design_action_dim = extra_design_action_dim

        self.voxels = np.arange(design_action_dim)  # [0,1,2,3,4] 
        self.voxels_prob = np.ones_like(self.voxels)/self.voxels.shape[0] # [0.2, 0.2, 0.2, 0.2, 0.2]
        self.mutated_idx_set = np.arange(granularity)
        self.species = []

        self.mutation_num = 1
        self.mutation_trial = 50
        self.initial_trial = 5

        self.species = [self.init_a_new_species() for _ in range(self.start_num_species)]
        self.gene_tree = {}

        self.granularity_error = np.zeros(self.wh**2)

        

    def update_probs(self, granularity):
        assert 1 <= granularity <= self.wh**2
        if granularity != self.granularity: 
            if granularity > self.granularity:
                self.mutated_idx_set = np.arange(self.granularity, granularity)
                
            elif granularity < self.granularity:
                self.mutated_idx_set = np.arange(granularity, self.granularity)

            self.granularity = granularity

            for i in range(len(self.species)): 
                self.species[i]['Count'] = 0  
                self.species[i]['AvgRwd'] = 0.0


    def sample_a_species(self):
        while True:
            i = np.random.choice(len(self.species))
            if not self.species[i]['InUse']:
                self.species[i]['InUse'] = True
                species = self.species[i]
                break
        return species
    
    def best_species(self, n):
        self.rank_species()
        count = 0
        cur_idx = len(self.species) - 1
        best_species = []
        while count < n and cur_idx >= 0:
            if self.species[cur_idx]['Count'] > 0:
                best_species.append(self.species[cur_idx])
                count += 1
            cur_idx -= 1
        
        if len(best_species) == n:
            return_species = best_species[::-1]
        else:
            return_species = self.species[-n:][::-1]
        
        print(f"best species: {[s['SpcID'] for s in return_species]}")
        return return_species 

    def update_a_species(self, species, design_failed=False):
        for i in range(len(self.species)): 
            if self.species[i]['SpcID'] == species['SpcID']:
                if design_failed:
                    self.species.pop(i)
                else:
                    alpha = 0.5 
                    self.species[i]['LastRwd'] = species['LastRwd']
                    self.species[i]['PredVal'] = species['PredVal']
                    self.species[i]['AvgRwd'] = self.species[i]['AvgRwd'] * alpha + species['LastRwd'] * (1 - alpha)
                    self.species[i]['AvgPredVal'] = self.species[i]['AvgPredVal'] * alpha + species['PredVal'] * (1 - alpha)
                    self.species[i]['InUse'] = False
                    self.species[i]['Count'] += 1
                    g = self.species[i]['Granularity']
                    delta = np.abs(self.species[i]['LastRwd'] - self.species[i]['AvgRwd'])
                    self.granularity_error[g - 1] =  self.granularity_error[g - 1] * alpha + delta * (1 - alpha)
                break

    def log_information(self):
        logging_items = ['SpcID', 'AvgRwd', 'LastRwd', 'PrtID', 'InUse', 'Count', 'Granularity']
        print(f"num_total_species: {self.num_total_species}")
        print(('| %10s ' * len(logging_items) + '|')
                    % tuple(logging_items))
        for s in self.species:
            print(f"| {s['SpcID']:10d} | {s['AvgRwd']:10.6f} | {s['LastRwd']:10.6f} | {s['PrtID']:10d} | {s['InUse']:10d} | {s['Count']:10d} | {s['Granularity']:10d} |")

    def natural_selection(self, disaster=False):
        self.current_generation += 1

        # get_rid of the worst species
        self.rank_species()

        if disaster:
            num_species_to_kill =  np.floor(len(self.species) * self.reset_elimination_rate)
        else:
            num_species_to_kill =  np.floor(len(self.species) * self.elimination_rate)
        
        print('Current species left in bank %d' % (len(self.species)))
        cur_eliminated = 0
        num_killed = 0
        for _ in range(int(num_species_to_kill)):
            if not self.species[cur_eliminated]['InUse'] and self.species[cur_eliminated]['Count'] > MIN_LIFE_LENGTH:  #4
                self.species.pop(cur_eliminated)  # rip, the eliminated species
                num_killed += 1
            else:
                cur_eliminated += 1
        print('After eliminating, current species left in bank %d' % (len(self.species)))

        # randomly perturb new species
        self.mutation_and_reproduction()
        print('After reproducing, current species left in bank %d' % (len(self.species)))

    def rank_species(self, sort_key='AvgRwd', consider_count=False):
        if consider_count:
            assert isinstance(sort_key, list)
            considered_species = [s for s in self.species if s['Count'] > 2] 
            chosen_len = 5
            if len(considered_species) >= chosen_len:
                chosen_species = considered_species
                weight = []
                for s1 in chosen_species:
                    w = []
                    for s2 in chosen_species:
                        w.append((s1['Design'] != s2['Design']).sum() / (self.wh**2))
                    weight.append(w)
                weight = np.array(weight)

                ranks = []
                for k in sort_key:
                    average_reward = [s[k] for s in chosen_species]
                    inverse_species_rank = np.argsort(average_reward)
                    ranks.append(inverse_species_rank)
                return (*ranks, weight)
            else:
                return (None,) * (len(sort_key) + 1)
        else:
            assert isinstance(sort_key, str)
            average_reward = [i_species_data[sort_key] for i_species_data in self.species]
            inverse_species_rank = np.argsort(average_reward)
            self.species = [self.species[i_rank] for i_rank in inverse_species_rank]
            return np.array([s['SpcID'] for s in self.species])


    def mutation_and_reproduction(self):
        # assert len(self.species) != 0, 'Oops! All the species died out.'
        if len(self.species) != 0:
            # when there is still species left, evolve the population based on the curent species_bank
            self.num_new_species = self.maximum_num_species - len(self.species)
            self.last_gen_num = len(self.species)
            print('New species to be added %d' % (self.num_new_species))

            for _ in range(self.num_new_species):
                # randomly choose the parent
                p_species = self.species[np.random.randint(self.last_gen_num)]
                if not p_species['InUse'] and p_species['Count'] > MIN_MATURE_LENGTH:
                    c_species =  self.mutate_species(p_species)
                    self.species.append(c_species)
        else:
            self.species = [self.init_a_new_species() for _ in range(self.start_num_species)]
        
    def mutate_species(self, p_species):
        c_species = dict()
        c_species['SpcID'] = self.num_total_species
        self.num_total_species += 1
        c_species['AvgRwd'] = p_species['AvgRwd']
        c_species['LastRwd'] = p_species['LastRwd']
        c_species['PredVal'] = p_species['PredVal']
        c_species['AvgPredVal'] = p_species['AvgPredVal']
        c_species['PrtID'] = p_species['SpcID']
        c_species['InUse'] = False
        c_species['Count'] = 0
        Design = p_species['Design'].copy()
        ExtraD = p_species['ExtraD'].copy()
        c_species['IndexInfo'] = index_info = p_species['IndexInfo']
        cur_index = deepcopy(p_species['CurIndex'])
        Granularity = p_species['Granularity']

        for m_num in range(self.mutation_num):
            # design type
            Design_copy = Design.copy()
            ExtraD_copy = ExtraD.copy()
            success = False
            for t_num in range(self.mutation_trial):
                if np.random.rand() < 0.8 or self.fixed_granularity:
                    mutation_rate=0.1                    
                    pd = get_uniform(5) 
                    pd[0] = 0.6 #it is 3X more likely for a cell to become empty
                    mutation = [mutation_rate, 1-mutation_rate] # for every cell there is mutation_rate% chance of mutation
                    for changed_idx in range(Granularity):
                        if draw(mutation) == 0: # mutation
                            changed_type = draw(pd)
                            Design[ExtraD == changed_idx] = changed_type
                        
                else:
                    Granularity_new = np.clip(Granularity + np.random.choice([-1,1,-2,2,-3,3]), 1, self.wh**2)
                    if Granularity_new > Granularity:
                        for ind in deepcopy(cur_index):                        
                            if len(index_info[ind]['child']) > 0:
                                cur_index.remove(ind)
                                cur_index.extend(index_info[ind]['child'])
                        
                    elif Granularity_new < Granularity:
                        for ind in deepcopy(cur_index):     
                            if index_info[ind]['parent'] is not None:
                                all_exist = [rem in cur_index for rem in index_info[index_info[ind]['parent']]['child']]   
                                if np.all(all_exist):                             
                                    for rem in index_info[index_info[ind]['parent']]['child']:
                                        cur_index.remove(rem)
                                    cur_index.append(index_info[ind]['parent'])

                    Granularity = len(cur_index)
                    ExtraD = np.zeros((self.wh, self.wh), dtype=int)
                    for i, ind in enumerate(cur_index):
                        for p in index_info[ind]['leaf']:
                            ExtraD[p//self.wh, p%self.wh] = i 
                    
                    for i in range(Granularity):
                        unique, counts = np.unique(Design[ExtraD == i], return_counts=True)
                        Design[ExtraD == i] = unique[counts.argmax()]
                    
                    
                if self.is_a_valid_design(Design, ExtraD):
                    success = True
                    break
            if not success:
                Design = Design_copy
                ExtraD = ExtraD_copy
                        
        c_species['Design'] = Design
        c_species['ExtraD'] = ExtraD
        c_species['Granularity'] = Granularity
        c_species['CurIndex'] = cur_index

        return c_species

    def init_a_new_species(self):
        while True:
        
            index2cluster = {
                i: [[i//self.wh, i%self.wh]] for i in range(self.wh**2)
            }
            index_parent = {
                i: None for i in range(self.wh**2)
            }

            while len(index2cluster) > 1:
                n = len(index2cluster)
                distance = np.zeros((n,n))

                for i in range(n):
                    for j in range(n):
                        a = np.array(index2cluster[i])
                        b = np.array(index2cluster[j])
                        d = np.expand_dims(a,axis=0) - np.expand_dims(b,axis=1)
                        distance[i,j] = distance[j,i] = np.sqrt((d**2).sum(axis=-1)).max()
                
                for i in range(n):
                    distance[i,i] = np.inf

                nearest = np.argwhere(distance == distance.min())
                i,j = nearest[np.random.choice(len(nearest))]
                
                index_i = index2cluster[i][0][0] * self.wh + index2cluster[i][0][1]
                index_j = index2cluster[j][0][0] * self.wh + index2cluster[j][0][1]
                while index_parent[index_i] is not None:
                    index_i = index_parent[index_i]
                while index_parent[index_j] is not None:
                    index_j = index_parent[index_j]
                index_parent[index_i] = index_parent[index_j] = len(index_parent)
                index_parent[len(index_parent)] = None
                
                index2cluster[i].extend(index2cluster[j])
                index2cluster.pop(j)

                index2cluster_new = dict()
                for i, v in enumerate(index2cluster.values()):
                    index2cluster_new[i] = v
                index2cluster = index2cluster_new
            
            index_child = {
                i: [] for i in range(self.wh**2)
            }
            for k,v in index_parent.items():
                if v is not None:
                    if v not in index_child:
                        index_child[v] = [k]
                    else:
                        index_child[v].append(k)

            index_leaf = {}
            for k, v in index_child.items():
                leaf = []
                def expand_leaf(ind):
                    for i in ind:
                        if len(index_child[i]) == 0:
                            leaf.append(i)
                        else:
                            expand_leaf(index_child[i])
                expand_leaf([k])
                index_leaf[k] = leaf

            index_info = {
                k: {
                    'parent': index_parent[k],
                    'child': index_child[k],
                    'leaf': index_leaf[k]
                }  for k in index_parent.keys() 
}

            if self.fixed_granularity:
                Granularity = self.wh**2
                cur_index = list(range(Granularity))
                ExtraD = np.arange(self.wh**2).reshape(self.wh, self.wh)
            else:
                Granularity = np.random.choice(range(1, self.wh**2+1))  
                
                cur_g = 1
                cur_index = [max(index_info)]
                
                while cur_g < Granularity:
                    for ind in deepcopy(cur_index):                        
                        if len(index_info[ind]['child']) > 0:
                            cur_index.remove(ind)
                            cur_index.extend(index_info[ind]['child'])
                    cur_g = len(cur_index)
                Granularity = len(cur_index)
                
                         
                ExtraD = np.zeros((self.wh, self.wh), dtype=int)
                for i, ind in enumerate(cur_index):
                    for p in index_info[ind]['leaf']:
                        ExtraD[p//self.wh, p%self.wh] = i 

            Design = np.zeros((self.wh, self.wh), dtype=int)
            for i in range(Granularity):
                Design[ExtraD == i] = np.random.choice(self.voxels, p=self.voxels_prob)
            
            if self.is_a_valid_design(Design, ExtraD):
                break
        
        species = {
            'SpcID': self.num_total_species,
            'AvgRwd': 0.,
            'LastRwd': 0.,
            'PredVal': 0.,
            'AvgPredVal': 0.,
            'PrtID': -1,
            'Design': Design,
            'ExtraD': ExtraD,
            'Granularity': Granularity,
            'IndexInfo': index_info,
            'CurIndex': cur_index,
            'InUse': False,
            'Count': 0,
        }
        self.num_total_species += 1
        return species

    def is_a_valid_design(self, design, extrad):
        return True 
        # return is_connected(design) and has_actuator(design)
