import networkx as nx
import numpy as np
import torch
import torch.utils.data
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
from collections import defaultdict
import os
from os import listdir
from os.path import join
from copy import deepcopy
import pickle
from .sarkar import sarkar
from .math import dist, expmap0, logmap0
from .optimizer import CEM

TRIAL_NUM = 10 

class NaiveCEMModel:
    def __init__(self, channel, width, height, depth, device=None):
        
        self.channel = channel
        self.width = width
        self.height = height
        self.depth = depth
        self.dim = self.channel * self.width * self.height * self.depth
        self.data_shape = (channel, width, height, depth)

        self.n_particle = self.width * self.height * self.depth
        self.latent_dim = self.dim
        self.device = device

        self.init_es()
        
    
    def init_es(self, init_mean = None):
        if init_mean is None:
            init_mean = np.zeros(self.latent_dim)

        self.hyperbolic_optimizer = CEM(init_mean, sigma_init=0.2, sigma_end=0.01, decay_iteration=3000,
                                        population_size = 10)
        self.hyperbolic_solutions = []
    
    def generate_from_euclidean(self, zs_euclidean, mean_action=False):
        zs = zs_euclidean.reshape(-1, self.channel, self.n_particle)
        robots = zs.argmax(1)
        components = torch.arange(self.n_particle).unsqueeze(0).repeat(robots.shape[0], 1).to(robots.device)
        return robots, components
    
    def ask(self):
        x_for_eval = x_for_tell = self.hyperbolic_optimizer.ask()
        return x_for_eval, x_for_tell

    def ask_mean(self):
        x_for_eval = x_for_tell = self.hyperbolic_optimizer._mean
        return x_for_eval, x_for_tell

    def tell(self, solution):
        self.hyperbolic_solutions.append(solution)
        if len(self.hyperbolic_solutions) >= self.hyperbolic_optimizer.population_size:
            self.hyperbolic_optimizer.tell(self.hyperbolic_solutions)
            self.hyperbolic_solutions = []

            if self.hyperbolic_optimizer.should_stop():
                print("STOP and RE-INIT!!!")
                self.init_es(self.hyperbolic_optimizer.ask())
    
    def update(self, iteration):
        if isinstance(self.hyperbolic_optimizer, CEM):
            self.hyperbolic_optimizer.update(iteration) 

class Manifold:
    def __init__(self, c=1.0):
        self.c=c
        
    def expmap0(self, u):
        return expmap0(u, c=self.c)
    
    def logmap0(self, u):
        return logmap0(u, c=self.c)
    
    def dist(self, x, y):
        return dist(x, y, c=self.c)


class HyperbolicModel:
    def __init__(self, channel, width, height, depth, numberOfChildren=2, device=None):
        
        self.channel = channel
        self.width = width
        self.height = height
        self.depth = depth
        self.dim = self.channel * self.width * self.height * self.depth
        self.data_shape = (channel, width, height, depth)
        self.root = np.zeros(self.dim)
        self.numberOfChildren = int(numberOfChildren)
        self.device = device

        self.n_particle = self.width * self.height * self.depth

        self.data, self.labels, self.comp, self.tree = self.bst()
        self.latent_dim = 2 
        self.curvature = 1.2 
        self.manifold = Manifold(c=self.curvature) 
        self.init_es()

        # initialize robots and latents
        self.Z = sarkar(self.tree, root=0, tau=self.curvature) 

        
        robots = np.stack([get_robot_from_probs(self.data[i].reshape(self.data_shape)) 
                           for i in range(len(self.data))])

        components = self.comp.reshape(-1, *self.data_shape[1:])
        self.Z_torch = torch.from_numpy(self.Z).to(device)[1:]
        self.data_torch = torch.from_numpy(self.data).to(device)[1:]
        self.labels_torch = torch.from_numpy(self.labels).to(device)[1:]
        self.levels = self.labels.argmin(axis=-1)
        self.levels_torch = self.labels_torch.argmin(dim=-1)
        self.cur_level = 1
        self.robot_torch = torch.from_numpy(robots).to(device)[1:]
        self.components_torch = torch.from_numpy(components).to(device)[1:]
        
        self.plot()


    def plot(self, save_path = None, log_hyperbolic_solutions = None):
        

        plt.figure(figsize=(6,6))
        ax = plt.gca()
        ax.add_patch(plt.Circle((0,0), 1.0/np.sqrt(1.0), color='grey',fill=False))
        
        if log_hyperbolic_solutions is not None:
            cur_ind = self.levels <= self.cur_level
            cur_edges = [e for e in self.tree.edges if self.levels[e[0]] <=self.cur_level and self.levels[e[1]] <=self.cur_level]
        else:
            cur_ind = np.full_like(self.levels, True, dtype=bool)
            cur_edges = self.tree.edges
        
        lines = [(self.Z[e[0]], self.Z[e[1]]) for e in cur_edges]
        lc = LineCollection(lines,linewidths=0.7,alpha=0.2, color='grey')
        ax.add_collection(lc)
        plt.scatter(self.Z[cur_ind,0], self.Z[cur_ind,1], s=15, c=self.labels[cur_ind].argmin(axis=1), cmap='PuBu', alpha=0.8, marker='.') 

        if log_hyperbolic_solutions is not None:
            X0 = np.array([x[0][0] for x in log_hyperbolic_solutions])
            X1 = np.array([x[0][1] for x in log_hyperbolic_solutions])
            reward_x = np.array([x[1] for x in log_hyperbolic_solutions])
            plt.scatter(X0, X1, c=reward_x, s=20, marker='x', cmap='viridis')

        ax.set_aspect("equal")
        plt.axis("off")

        if save_path is None:
            save_path = os.path.join("data", f"embedding.png")

        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=600, transparent=False)
        plt.close()

    def __len__(self):
        '''
        this method returns the total number of samples/nodes
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Generates one sample
        '''
        data, labels = self.data[idx], self.labels[idx]
        return torch.Tensor(data), torch.Tensor(labels)

    def get_data(self):
        return self.data, self.labels, self.tree    
    
    def init_es(self, init_mean = None):
        if init_mean is None:
            init_mean = np.zeros(self.latent_dim)

        self.hyperbolic_optimizer = CEM(init_mean, sigma_init=0.2, sigma_end=0.01, decay_iteration=3000,
                                        population_size = 10) 

        self.hyperbolic_solutions = []

    def get_nearest_idx(self, zs_euclidean, return_new_zs=False):
        zs = self.manifold.expmap0(zs_euclidean)
        cur_ind = self.levels_torch <= self.cur_level
        distance = self.manifold.dist(zs.unsqueeze(1), self.Z_torch[cur_ind].unsqueeze(0))
        nearest_idx = distance.argmin(dim=-1)
        if return_new_zs:
            zs_hyperbolic = self.Z_torch[cur_ind][nearest_idx]
            zs_euclidean_new = self.manifold.logmap0(zs_hyperbolic)
            return zs_euclidean_new
        else:
            return nearest_idx
    
    def generate_from_euclidean(self, zs_euclidean, mean_action=False):
        nearest_idx = self.get_nearest_idx(zs_euclidean)
        robots = self.robot_torch[nearest_idx]
        components = self.components_torch[nearest_idx]
        if mean_action and self.levels_torch[nearest_idx].max() >= self.cur_level: 
            self.cur_level += 1
            print(f"increase current level: {self.cur_level}")
        return robots, components
    
    def ask(self):
        x_for_eval = x_for_tell = self.hyperbolic_optimizer.ask()
        return x_for_eval, x_for_tell

    def ask_mean(self):
        x_for_eval = x_for_tell = self.hyperbolic_optimizer._mean
        return x_for_eval, x_for_tell

    def tell(self, solution):
        self.hyperbolic_solutions.append(solution)
        if len(self.hyperbolic_solutions) >= self.hyperbolic_optimizer.population_size:
            self.hyperbolic_optimizer.tell(self.hyperbolic_solutions)
            self.hyperbolic_solutions = []

            if self.hyperbolic_optimizer.should_stop():
                print("STOP and RE-INIT!!!")
                self.init_es(self.hyperbolic_optimizer.ask())
    
    def update(self, iteration):
        if isinstance(self.hyperbolic_optimizer, CEM):
            self.hyperbolic_optimizer.update(iteration) 


    def bst(self):
        self.index_mapping = {}
        save_path = 'data'
        data_path = join(save_path,'train.p')
        cluster_path = join(save_path, 'cluster.p')

        if os.path.exists(data_path):
            print("====== Loading Training Dataset ======")
            values_visited, labels_visited, comp_visited, tree = pickle.load(open(data_path, 'rb'))
        else:
            print("====== Generating Training Dataset ======")
            all_n_clusters = [1,2,4] 
            ## generate clusters
            if os.path.exists(cluster_path):
                cluster_data = pickle.load(open(cluster_path, 'rb'))
            else:
                index2cluster = {
                    i: [[i]] for i in range(self.n_particle)
                }
                index_parent = {
                    i: None for i in range(self.n_particle)
                }
                cluster_nums = deepcopy(sorted(all_n_clusters, reverse=True))
                
                index_xyz = np.arange(self.n_particle).reshape(self.depth, self.width, self.height).transpose([1,2,0]).flatten()
                self.particle_xyz = np.zeros((self.n_particle, 3))
                self.particle_xyz[:, 0] = index_xyz % self.width
                self.particle_xyz[:, 1] = (index_xyz // self.width) % self.height
                self.particle_xyz[:, 2] = (index_xyz // self.width) // self.height

                cluster_data = {}

                while len(index2cluster) > 1 and len(cluster_nums) > 0:
                    n_cluster = cluster_nums.pop(0) 
                    xyz = np.stack([
                        self.particle_xyz[np.array(index2cluster[i]).squeeze(-1)].mean(0) for i in range(len(index2cluster))
                    ], axis=0)

                    cluster_label, cluster_center = get_cluster_map(xyz, n_cluster)
                    true_cluster_label = np.zeros(self.n_particle, dtype=cluster_label.dtype)
                    for i in range(cluster_label.shape[0]):
                        true_cluster_label[np.concatenate(index2cluster[i])] = cluster_label[i]
                    neighboring_label = {}
                    for l in np.unique(true_cluster_label):
                        loc = self.particle_xyz[true_cluster_label == l]
                        neighbor_loc = np.concatenate([
                            loc + shift for shift in [
                                np.array([[1, 0, 0]]), np.array([[-1, 0, 0]]), 
                                np.array([[0, 1, 0]]), np.array([[0, -1, 0]]), 
                                np.array([[0, 0, 1]]), np.array([[0, 0, -1]]), 
                            ]
                        ])
                        neighbor_loc = neighbor_loc.clip([0, 0, 0],[self.width-1, self.height-1, self.depth-1])
                        neighbor_loc_index = neighbor_loc[:, 0] + neighbor_loc[:, 1] * self.width +\
                                             neighbor_loc[:, 2] * self.width * self.height
                        neighbor_loc_label = np.unique(true_cluster_label[neighbor_loc_index.astype(int)])
                        neighbor_loc_label = np.delete(neighbor_loc_label, np.where(neighbor_loc_label == l))
                        neighboring_label[l] = np.array([ll in neighbor_loc_label for ll in true_cluster_label])

                    
                    cluster_data[f'labels_{n_cluster}'] = true_cluster_label
                    cluster_data[f'centers_{n_cluster}'] = cluster_center
                    cluster_data[f'neighbor_{n_cluster}'] = neighboring_label

                    for label in range(n_cluster):
                        cur_i_list = np.where(cluster_label == label)[0]
                        for i in cur_i_list:
                            index_i = index2cluster[i][0][0]
                            while index_parent[index_i] is not None:
                                index_i = index_parent[index_i]
                            
                            index_parent[index_i] = len(index_parent) # set parent
                            if i != cur_i_list[0]:
                                index2cluster[cur_i_list[0]].extend(index2cluster[i])
                                index2cluster.pop(i)                    

                        index_parent[len(index_parent)] = None # add a new parent

                    index2cluster_new = dict()
                    for i, v in enumerate(index2cluster.values()):
                        index2cluster_new[i] = v
                    index2cluster = index2cluster_new

                index_child = {
                    i: [] for i in range(self.n_particle)
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

                self.index_info = {
                    k: {
                        'parent': index_parent[k],
                        'child': index_child[k],
                        'leaf': index_leaf[k],
                    }  for k in index_parent.keys() 
                }

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                pickle.dump(cluster_data, open(cluster_path, 'wb'))

            values_visited, labels_visited, comp_visited, tree = tree_generation(all_n_clusters, cluster_data, self.channel, self.n_particle, self.data_shape,
                                                             n_data_points=5000000, numberOfChildren=8) 
                    
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pickle.dump([values_visited, labels_visited, comp_visited, tree], open(data_path, 'wb'))
        
        print(f"Training Dataset Size: {values_visited.shape[0]}")
        print("====== End of Generating Training Dataset ======")
        return values_visited, labels_visited, comp_visited, tree




def get_cluster_map(xyz_np, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters).fit(xyz_np)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return labels, centers

def tree_generation(all_n_clusters, 
                    cluster_data, 
                    channel,
                    n_particle,
                    data_shape,
                    n_data_points = 5000,
                    numberOfChildren = 2):
    
    ## generate data points
    SIGMA = 1.0
    index = 0
    max_depth = len(all_n_clusters)  - 1

    queue = []
    count_samples_per_depth = defaultdict(lambda:0)
    cur_depth = -1
    cur_value = np.zeros((channel, n_particle)).astype(np.float32)
    cur_label = np.zeros(max_depth + 1 + 1)
    cur_comp = np.zeros(n_particle).astype(np.int32)
    queue.append((cur_value, cur_label, cur_depth, index))

    values_visited = [cur_value] # robots
    labels_visited = [cur_label] # labels
    comp_visited = [cur_comp] # components 

    hashes_visited = {get_hash_code(get_robot_from_probs(cur_value.reshape(data_shape)))}
    T = nx.Graph()
    T.add_node(0)
    while len(queue) > 0 and index < n_data_points: # expand the tree
        par_value, par_label, par_depth, par_index = queue.pop(0)
        cur_depth = par_depth + 1
        par_robot = get_robot_from_probs(par_value.reshape(data_shape))
       
        if par_depth < max_depth:
            par_cluster_label = cluster_data[f'labels_{all_n_clusters[par_depth]}']
            cur_cluster_label = cluster_data[f'labels_{all_n_clusters[cur_depth]}']
            true_c = 0
            for c in range(numberOfChildren): 
                success = False
                for _ in range(TRIAL_NUM): 
                    cur_value = par_value.copy()
                    for ind in np.unique(cur_cluster_label):
                        cur_index = (cur_cluster_label == ind)
                        cur_value = set_value(cur_value, np.random.choice(channel), cur_index, SIGMA)
                    cur_robot = get_robot_from_probs(cur_value.reshape(data_shape))
                    if is_valid(cur_value.reshape(data_shape)) \
                        and (cur_depth ==0 or is_similar(par_robot, cur_robot, par_cluster_label)) \
                        and get_hash_code(cur_robot) not in hashes_visited:
                        success = True
                        break
                    # success = True
                    # break
                
                if success:
                    cur_label = par_label.copy()
                    cur_label[cur_depth] = true_c + 1
                    true_c += 1

                    values_visited.append(cur_value)
                    labels_visited.append(cur_label)
                    comp_visited.append(cur_cluster_label)
                    hashes_visited.update({get_hash_code(cur_robot)})
                    index += 1
                    T.add_node(index)
                    T.add_edge(par_index, index) 

                    queue.append((cur_value, cur_label, cur_depth, index))

                    count_samples_per_depth[cur_depth] += 1

                    print(f"{all_n_clusters[cur_depth]}:{cur_label}\n{cur_robot.squeeze(-1)}") 
    
    print(f"left in queue: {len(queue)}")
    print(f"#samples per depth: {count_samples_per_depth}")
    a=[get_robot_from_probs(values_visited[i].reshape(data_shape)) for i in range(len(values_visited))]
    b = np.unique(a, axis=0)
    print(f"unique robots in dataset: {len(b)} / {len(a)}")

    values_visited = np.stack(values_visited)
    values_visited = values_visited.reshape(values_visited.shape[0], -1)
    labels_visited = np.stack(labels_visited)
    comp_visited = np.stack(comp_visited)
    return values_visited, labels_visited, comp_visited, T


def set_value(v, which_channel, index, sigma):

    random_value = np.zeros((v.shape[0], 1))
    p = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    p /= p.sum()
    random_value[np.random.choice(range(5), p=p)] = 1.0
    v[:, index] = v[:, index] * 0.0 + np.repeat( 
        random_value, index.sum(), axis=1) * np.sqrt(sigma)

    return v

def get_robot_from_probs(v):
    assert len(v.shape) == 4 # channel, width, height, depth
    return v.argmax(0)

def is_valid(v):
    channel, width, height, depth = v.shape
    v = get_robot_from_probs(v)

    visited = np.full_like(v, False)
    locs = [(i,j,k) for i in range(width) for j in range(height) for k in range(depth)]
    dxyz = [(i,j,k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1] if abs(i)+abs(j)+abs(k) <=1]
    
    for (i,j,k) in locs:
        if v[i, j, k] != 0:
            break
    
    def visit(x, y, z, item):
        if 0<= x <width and 0<= y <height and 0<= z <depth \
            and v[x,y,z] != 0 \
            and not visited[x,y,z]:
            visited[x,y,z] = True
            for (dx,dy,dz) in dxyz:
                visit(x+dx,y+dy,z+dz,item)
    
    visit(i,j,k, v[i,j,k])
    is_connected = (visited.sum() == (v != 0).sum())

    has_actuator = ((v == 3).sum() + (v == 4).sum()) > 0

    return (is_connected and has_actuator)


def is_similar(par_robot, cur_robot, par_cluster_label):
    for ind in np.unique(par_cluster_label):
        par_unique, par_counts = np.unique(par_robot.flatten()[par_cluster_label == ind], return_counts=True)
        cur_unique, cur_counts = np.unique(cur_robot.flatten()[par_cluster_label == ind], return_counts=True)
        if len(cur_unique) == 2: 
            if not np.isclose(np.expand_dims(cur_unique, 1), np.expand_dims(par_unique,0)).any():
                return False
            
        else:

            if not np.isclose(np.expand_dims(cur_unique[cur_counts == cur_counts.max()], 1),  
                              np.expand_dims(par_unique[par_counts == par_counts.max()], 0)).any():
                return False
    return True


def cal_distance(par_value, cur_value):
    return ((cur_value - par_value)**2).mean()

def get_n_component(v):
    channel, width, height, depth = v.shape
    v = get_robot_from_probs(v)

    visited = np.full_like(v, False)
    locs = [(i,j,k) for i in range(width) for j in range(height) for k in range(depth)]
    dxyz = [(i,j,k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]]
    n_component = 0
    while not visited.all():
        n_component += 1        
        for (i,j,k) in locs:
            if not visited[i,j,k]:
                break
        
        def visit(x, y, z, item):
            if 0<= x <width and 0<= y <height and 0<= z <depth \
                and np.isclose(v[x,y,z],item) \
                and not visited[x,y,z]:
                visited[x,y,z] = True
                for (dx,dy,dz) in dxyz:
                    visit(x+dx,y+dy,z+dz,item)
        
        visit(i,j,k, v[i,j,k])
        
    return n_component

def get_hash_code(v):
    return hash(str(v))


def get_component_labels(v):
    channel, width, height, depth = v.shape
    v = get_robot_from_probs(v)

    visited = np.full_like(v, False)
    labels = np.full_like(v, np.inf)
    locs = [(i,j,k) for i in range(width) for j in range(height) for k in range(depth)]
    dxyz = [(i,j,k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]]
    n_component = 0
    while not visited.all():
        n_component += 1        
        for (i,j,k) in locs:
            if not visited[i,j,k]:
                break
        
        def visit(x, y, z, item):
            if 0<= x <width and 0<= y <height and 0<= z <depth \
                and np.isclose(v[x,y,z],item) \
                and not visited[x,y,z]:
                visited[x,y,z] = True
                labels[x,y,z] = n_component - 1
                for (dx,dy,dz) in dxyz:
                    visit(x+dx,y+dy,z+dz,item)
        
        visit(i,j,k, v[i,j,k])
    
    return labels

class FixedModel:
    def __init__(self, channel, width, height, depth, device=None):
        
        ## learned
        # self.act = torch.tensor([
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 0, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        # ], dtype=torch.int32, device=device)
        # self.extra_act = torch.tensor([
        #     [1, 1, 1, 3, 3],
        #     [1, 1, 1, 3, 3],
        #     [2, 2, 0, 3, 3],
        #     [2, 2, 0, 0, 0],
        #     [2, 2, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        #
        # self.act = torch.tensor([
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        # ], dtype=torch.int32, device=device)
        # self.extra_act = torch.tensor([
        #     [0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # 
        # self.act = torch.tensor([
        #     [4, 4, 4, 0, 0],
        #     [4, 4, 4, 0, 0],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)
        # self.extra_act = torch.tensor([
        #     [1, 1, 1, 0, 0],
        #     [1, 1, 1, 0, 0],
        #     [1, 1, 0, 0, 0],
        #     [1, 1, 0, 0, 0],
        #     [1, 1, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # 
        # self.act = torch.tensor([
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 0, 4, 4],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)
        # self.extra_act = torch.tensor([
        #     [1, 1, 1, 3, 3],
        #     [1, 1, 1, 3, 3],
        #     [2, 2, 0, 3, 3],
        #     [2, 2, 0, 0, 0],
        #     [2, 2, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        ## handcrafted
        self.act = torch.tensor([
            [3, 3, 2, 3, 3],
            [3, 3, 2, 3, 3],
            [4, 4, 0, 4, 4],
            [4, 4, 0, 4, 4],
            [4, 4, 0, 4, 4],
        ], dtype=torch.int32, device=device)
        self.extra_act = torch.tensor([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
        ], dtype=torch.int32, device=device)

        ## GA
        # self.act = torch.tensor([
        #     [3, 0, 0, 0, 2,],
        #     [4, 1, 4, 1, 2,],
        #     [2, 3, 3, 4, 3,],
        #     [4, 0, 2, 2, 3,],
        #     [1, 3, 0, 2, 3,],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [3, 3, 0, 0, 4,],
        #     [4, 1, 4, 1, 2,],
        #     [1, 3, 3, 0, 3,],
        #     [4, 0, 1, 2, 3,],
        #     [1, 2, 0, 2, 1,],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [3, 1, 0, 2, 0,],
        #     [4, 3, 0, 1, 1,],
        #     [0, 2, 0, 4, 0,],
        #     [4, 1, 2, 4, 0,],
        #     [1, 3, 0, 4, 4,],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [3, 1, 0, 2, 0,],
        #     [4, 3, 0, 1, 1,],
        #     [0, 3, 3, 4, 0,],
        #     [4, 1, 3, 4, 0,],
        #     [1, 3, 0, 4, 4,],
        # ], dtype=torch.int32, device=device)

        # ## CuCo
        # self.act = torch.tensor([
        #     [0, 0, 0, 0, 0,],
        #     [0, 0, 0, 0, 0,],
        #     [2, 3, 4, 0, 0,],
        #     [2, 3, 4, 0, 0,],
        #     [2, 4, 4, 0, 0,],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [2, 3, 3, 3, 4,],
        #     [2, 3, 3, 3, 4,],
        #     [2, 3, 3, 3, 4,],
        #     [2, 3, 3, 3, 4,],
        #     [2, 4, 4, 4, 4,],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [2, 3, 4, 3, 4,],
        #     [2, 3, 4, 3, 4,],
        #     [2, 4, 4, 3, 4,],
        #     [2, 3, 3, 3, 4,],
        #     [2, 4, 4, 4, 4,],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [1, 1, 1, 3, 4,],
        #     [1, 1, 2, 1, 4,],
        #     [2, 4, 4, 1, 4,],
        #     [2, 1, 2, 2, 4,],
        #     [2, 4, 4, 4, 4,],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [2, 3, 4, 3, 4,],
        #     [2, 1, 4, 2, 4,],
        #     [2, 4, 4, 1, 4,],
        #     [2, 1, 2, 2, 4,],
        #     [2, 4, 4, 4, 4,],
        # ], dtype=torch.int32, device=device)

        # for plot
        # self.act = torch.tensor([ 
        #     [3, 3, 3, 3, 3],
        #     [3, 3, 3, 3, 3],
        #     [3, 3, 3, 3, 3],
        #     [3, 3, 3, 3, 3],
        #     [3, 3, 3, 3, 3],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [3, 3, 3, 0, 0],
        #     [3, 3, 3, 0, 0],
        #     [3, 3, 0, 0, 0],
        #     [3, 3, 0, 0, 0],
        #     [3, 3, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [0, 0, 0, 3, 3],
        #     [0, 0, 0, 3, 3],
        #     [0, 0, 3, 3, 3],
        #     [0, 0, 3, 3, 3],
        #     [0, 0, 3, 3, 3],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 0, 0],
        #     [4, 4, 4, 0, 0],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 4, 4, 4],
        #     [0, 0, 4, 4, 4],
        #     [0, 0, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 0, 4, 4],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 3, 3],
        #     [4, 4, 4, 3, 3],
        #     [4, 4, 0, 3, 3],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 2, 2],
        #     [4, 4, 4, 2, 2],
        #     [4, 4, 0, 2, 2],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 0],
        #     [4, 4, 0, 4, 0],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 3],
        #     [4, 4, 0, 4, 3],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 3],
        #     [4, 4, 0, 4, 2],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 2],
        #     [4, 4, 0, 4, 2],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 2, 2],
        #     [4, 4, 4, 2, 2],
        #     [4, 4, 2, 2, 2],
        #     [4, 4, 2, 2, 2],
        #     [4, 4, 2, 2, 2],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [3, 3, 3, 4, 4],
        #     [3, 3, 3, 4, 4],
        #     [3, 3, 4, 4, 4],
        #     [3, 3, 4, 4, 4],
        #     [3, 3, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 2],
        #     [4, 4, 0, 2, 2],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 3],
        #     [4, 4, 0, 3, 3],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 1],
        #     [4, 4, 0, 1, 1],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 0],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 2, 2],
        #     [4, 4, 4, 2, 4],
        #     [4, 4, 0, 4, 4],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [4, 4, 4, 3, 3],
        #     [4, 4, 4, 3, 4],
        #     [4, 4, 0, 4, 4],
        #     [4, 4, 0, 0, 0],
        #     [4, 4, 0, 0, 0],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [3, 3, 3, 1, 1],
        #     [3, 3, 3, 1, 1],
        #     [3, 3, 1, 1, 1],
        #     [3, 3, 1, 1, 1],
        #     [3, 3, 1, 1, 1,]
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [2, 2, 2, 2, 2],
        #     [2, 2, 2, 2, 2],
        #     [2, 2, 3, 3, 3],
        #     [3, 3, 3, 3, 3],
        #     [3, 3, 3, 3, 3,]
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([ 
        #     [0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0],
        #     [0, 0, 3, 3, 3],
        #     [3, 3, 3, 3, 3],
        #     [3, 3, 3, 3, 3,]
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [3, 3, 3, 3, 3],
        #     [3, 3, 3, 3, 3],
        #     [3, 3, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [1, 1, 1, 1, 1],
        #     [1, 1, 1, 1, 1],
        #     [1, 1, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [2, 2, 2, 4, 4],
        #     [2, 2, 2, 4, 4],
        #     [2, 2, 4, 4, 4],
        #     [2, 2, 4, 4, 4],
        #     [2, 2, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 0, 4, 4],
        #     [2, 2, 4, 4, 4],
        #     [2, 2, 4, 4, 4],
        #     [2, 2, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [1, 1, 1, 4, 4],
        #     [1, 1, 1, 4, 4],
        #     [0, 0, 4, 4, 4],
        #     [0, 0, 4, 4, 4],
        #     [0, 0, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 0, 4, 4],
        #     [3, 3, 4, 4, 4],
        #     [3, 3, 4, 4, 4],
        #     [3, 3, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [0, 0, 0, 2, 2],
        #     [0, 0, 0, 2, 2],
        #     [0, 0, 4, 2, 2],
        #     [0, 0, 4, 4, 4],
        #     [0, 0, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 3, 4, 4],
        #     [0, 0, 3, 3, 3],
        #     [0, 0, 3, 3, 3],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 0, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 3, 4, 4, 4],
        #     [3, 3, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 0, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 3, 3],
        #     [4, 4, 4, 3, 3],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 0, 4, 4],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 2, 2],
        #     [4, 4, 4, 2, 2],
        # ], dtype=torch.int32, device=device)

        # self.act = torch.tensor([
        #     [0, 0, 0, 4, 4],
        #     [0, 0, 0, 4, 1],
        #     [4, 4, 4, 1, 1],
        #     [4, 4, 4, 4, 4],
        #     [4, 4, 4, 4, 4],
        # ], dtype=torch.int32, device=device)

        

    def get_design(self):
        return self.act, self.extra_act



if __name__ == '__main__':

    data_shape = (5,5,5,1)
    channel = data_shape[0]

    model = HyperbolicModel(*data_shape, device='cuda')
