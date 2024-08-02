import numpy as np
import copy
import gym
from gym import spaces
from evogym import *
from evogym.envs import BenchmarkBase
from evogym.utils import _recursive_search
import os

# ModularEvoGym Wrapper
class modular_env(BenchmarkBase):
    def __init__(self, env, body=None, mode="modular") -> None:
        self.env = env
        self.env_id = self.env.spec.id
        self.init_position = (self.env.world.objects['robot'].pos.x, self.env.world.objects['robot'].pos.y)
        assert mode == "modular"
        body = np.zeros_like(body, dtype=int) 
        self.set_modular_attributes(body=body,mode=mode)

    # Set attributes
    def set_modular_attributes(self, body=None, mode="modular"):
        self.mode = mode
        self.stage = 'design'

        self.env.reset()
        self.world = self.env.world
        BenchmarkBase.__init__(self, self.world) 
        self.design_steps = 1 
        self.stage = 'design'
        self.cur_t = 0

        self.wh = wh = body.shape[0] 
        robot = np.expand_dims(body, axis=0)

        self.init_design = copy.deepcopy(robot) 
        self.cur_design = copy.deepcopy(robot) 


        self.im_size = robot.shape[-1] + 2
        self.number_neighbors = 9 # or 5 

        self.design_type_num = 5
        self.design_action_dim = self.design_type_num
        self.extra_design_action_dim = 25 
        
        #                                            neighbor type              loc               time step
        self.design_state_dim = self.number_neighbors * self.design_type_num + self.wh * 2 # + (self.design_steps + 1)

        self.loc_one_hot = np.eye(self.wh)
        self.type_one_hot = np.eye(self.design_type_num)
        self.timestep_ont_hot = np.eye(self.design_steps + 1)

        self.body = body
        self.voxel_num = self.body.size
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)
        self.obs_design = self.make_design_batch(self.cur_design)
        self.design_cluster = np.zeros((self.wh**2, self.wh**2))
        self.design_cluster_aug = np.zeros((self.wh**2, self.wh**2))

        self.init_modular_obs = self.get_modular_obs()
        self.observation_space = self.convert_obs_to_space(self.init_modular_obs)
        
        self.init_modular_obs["act_mask"] = np.zeros_like(self.init_modular_obs["act_mask"]) 
        self.init_modular_obs["obs_mask"] = np.zeros_like(self.init_modular_obs["obs_mask"])
        self.init_modular_obs['design'] = self.obs_design
        self.init_modular_obs['cluster'] = self.design_cluster.flatten()
        self.init_modular_obs['cluster_aug'] = self.design_cluster_aug.flatten()
        self.init_modular_obs['stage'] = [0.0]
        
        self.modular_state_dim = self.observation_space['modular'].shape[0] // self.voxel_num
        self.modular_action_dim = 1
        self.other_dim = self.observation_space['other'].shape[0]
    
    # Reset
    def reset(self):
        self.stage = 'design'
        self.cur_t = 0
        self.cur_design = self.init_design
        ob = self.init_modular_obs
        self.env.reset()
        return ob

    # Step
    def step(self, action):
        self.cur_t = self.cur_t + 1
        if self.stage == 'design':
            ob, reward, done, info = self.design_step(action)
            return ob, reward, done, info
        else:
            action = action[~self.act_mask.astype(bool)] # will be rescale to [0.6,1.6]
            _, reward, done, _ = self.env.step(action) 
            obs = self.get_modular_obs()
            return obs, reward, done, {'design_success': True, 'stage': 'act', 'act_step': self.cur_t - self.design_steps}

    def design_step(self, action):
        # fake modular obs
        ob = self.init_modular_obs
        self.cur_design, success, extra_act = self.apply_design_action(output=action, robot_seed=self.cur_design)
        ob['design'] = self.make_design_batch(self.cur_design) 
        ob['cluster'] = self.design_cluster.flatten()
        ob['cluster_aug'] = self.design_cluster_aug.flatten()

        if self.cur_t == self.design_steps:
            if not success: 
                design_penalty = 0.0 
                return ob, design_penalty, True, {'design_success': False, 'stage': 'design'} # obs is used for next episode, will automatically reset.
            
            self.stage = 'act'
            self.act_stage_design = self.cur_design
            self.real_design = self.cur_design[0] # original: self.cur_design[0,1:self.im_size-1,1:self.im_size-1] 
            self.update(body=self.real_design,position=self.init_position, env_id=self.env_id, 
                                 connections=get_full_connectivity(self.real_design))
            act_ob = self.get_modular_obs()
            act_ob['design'] = self.make_design_batch(self.act_stage_design)
            act_ob['cluster'] = self.design_cluster.flatten()
            act_ob['cluster_aug'] = self.design_cluster_aug.flatten()
            # act_ob['stage'] = [1.0]
            return act_ob, 0.0, False, {'design_success': 'Done', 'stage': 'design','real_design':self.real_design,
                                        'extra_act': extra_act, }
        reward = 0.0
        done = False
        return ob, reward, done, {'design_success': True, 'stage': 'design'}

    def apply_design_action(self, output, robot_seed=None):
        output = output.reshape(self.wh, self.wh).astype(int)
        type_act = output % self.design_type_num
        extra_act = output // self.design_type_num
        self.design_cluster = np.zeros((self.wh**2, self.wh**2))
        self.design_cluster_aug = np.zeros((self.wh**2, self.wh**2))
        extra_act_aug = extra_act.copy()
        
        if extra_act_aug.max() < self.wh**2 - 1:
            unique, counts = np.unique(extra_act_aug, return_counts=True)
            divided_index = unique[counts.argmax()]
            chosen_extra_act = (extra_act_aug == divided_index)
            extra_act_aug[chosen_extra_act] = np.random.choice([divided_index, extra_act_aug.max() + 1], 
                                                            size=np.sum(chosen_extra_act))
        
        robot_seed_tmp = np.zeros((1, self.wh, self.wh))
        for i in range(self.wh): 
            for j in range(self.wh):
                label1 = extra_act[i, j]
                type1 = type_act[i, j]
                robot_seed_tmp[0, i, j] = type1
                self.design_cluster[label1].reshape(self.wh, self.wh)[i,j] = 1.0/np.sum(np.isclose(extra_act, label1))
                
                label2 = extra_act_aug[i, j]
                self.design_cluster_aug[label2].reshape(self.wh, self.wh)[i,j] = 1.0/np.sum(np.isclose(extra_act_aug, label2))
        
        # check 
        output = copy.deepcopy(robot_seed_tmp.reshape(-1))

        robot_seed_tmp = copy.deepcopy(robot_seed)
        non0 = output.nonzero()[0]
        is3 = np.isclose(output, 3).nonzero()[0]
        is4 = np.isclose(output, 4).nonzero()[0]        
        min3 = is3.min() if is3.shape[0] > 0 else self.wh**2
        min4 = is4.min() if is4.shape[0] > 0 else self.wh**2
        if is3.shape[0] > 0 or is4.shape[0] > 0:
            start = min(min3, min4)
            start_i = start // self.wh
            start_j = start % self.wh
            output = output.reshape(self.wh, self.wh)
            connectivity = np.zeros_like(output)
            _recursive_search(start_i, start_j, connectivity, output)
            robot_seed_tmp[0] = output

        robot_seed = copy.deepcopy(robot_seed_tmp)        
        real_robot = robot_seed[0]
        
        if (is3.shape[0] > 0 or is4.shape[0] > 0) and is_connected(real_robot) and has_actuator(real_robot):
            success = True
        else:
            success = False
        return robot_seed, success, extra_act

    def update(self, position, env_id, body, connections):
        self.env = gym.make(self.env_id, body=body, connections=connections)
        self.env.reset() 
        self.world = self.env.world
        BenchmarkBase.__init__(self, self.world)
        super().reset()
        

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float) 
        self.body = body
        self.voxel_num = self.body.size
        self.obs_index_array, self.voxel_corner_size = self.transform_to_modular_obs(body=self.body)
        self.act_mask = self.get_act_mask(self.body)
        self.obs_mask = self.get_obs_mask(self.body)
        self.obs_design = self.make_design_batch(self.cur_design)        

    # Functions for creating state-action masks
    def get_act_mask(self, body):
        am = copy.deepcopy(body)
        actuator_masks = am.flatten()
        for i in range(len(actuator_masks)): 
            if actuator_masks[i].astype(int) == 4 or actuator_masks[i].astype(int) == 3:
                actuator_masks[i]=0
            else:
                actuator_masks[i]=1
        return actuator_masks.astype(int) 
    
    def get_obs_mask(self, body):
        bd = copy.deepcopy(body)
        bd_materials = bd.flatten()
        obs = []
        obs_padding = []

        for i in range(len(bd_materials)): 
            if bd_materials[i].astype(int) == 0:
                obs.append(1)
            else:
                obs.append(0)
        return np.append(obs, np.array(obs_padding)) 
    
    # Obs functions
    def get_modular_obs(self):
        obs = {}
        origin_ob=self.env.get_relative_pos_obs_nof("robot")
        obs["modular"]= self.modular_ob_wrapper_padding(origin_ob=origin_ob,index_array=self.obs_index_array,body=self.body)
        obs["other"] = self.get_other_obs()
        obs["act_mask"] = self.act_mask
        obs["obs_mask"] = self.obs_mask
        obs['stage'] = np.array([1.0])
        obs['design'] = self.obs_design 
        obs['cluster'] = self.design_cluster.flatten()
        obs['cluster_aug'] = self.design_cluster_aug.flatten()
        return obs

    def modular_ob_wrapper_padding(self, origin_ob, index_array, body):
        modular_obs_tmp = []
        modular_obs = []
        obs_padding = []
        body = list(body.flatten())
        for index in index_array:
            if np.sum(index) == -4:
                modular_obs_tmp.append(np.zeros(8).astype(float))
            else:
                modular_obs_tmp.append(np.concatenate((origin_ob[0][index],origin_ob[1][index]),axis=0))
        
        # Add material information
        # ps: If you do not want to add this information, just comment out this 'for' loop 

        for i in range(len(modular_obs_tmp)):
            modular_obs_tmp[i] = np.concatenate([modular_obs_tmp[i], self.type_one_hot[body[i]],
                                                self.loc_one_hot[i//self.wh], self.loc_one_hot[i%self.wh]])
            modular_obs.append(modular_obs_tmp[i]) 

        del modular_obs_tmp
        return np.array(modular_obs).flatten()    

    def convert_obs_to_space(self, observation):
        from collections import OrderedDict
        import numpy as np
        from gym import spaces
        if isinstance(observation, dict):
            space = spaces.Dict(
                OrderedDict(
                    [
                        (key, self.convert_obs_to_space(value))
                        for key, value in observation.items()
                    ]
                )
            )
        elif isinstance(observation, np.ndarray):
            low = np.full(observation.shape, -100.0, dtype=np.float32)
            high = np.full(observation.shape, 100.0, dtype=np.float32)
            space = spaces.Box(low, high, dtype=observation.dtype)
        else:
            raise NotImplementedError(type(observation), observation)
        return space

    def make_design_batch(self, state=None):
        robot_seed = state
        # Make batch
        batch_inputs = np.zeros((self.voxel_num, self.design_state_dim))
        counter = 0
        robot_seed = np.pad(robot_seed, ((0,0),(1,1),(1,1))) # original: no pad
        for i in range(1,self.im_size-1):
            for e in range(1,self.im_size-1):
                if self.number_neighbors == 9:
                    cell_input = robot_seed[0, i-1:i+2, e-1:e+2].flatten()
                elif self.number_neighbors == 5:
                    cell_input = np.concatenate([robot_seed[0, i-1, e],robot_seed[0, i+1, e],robot_seed[0, i, e+1],robot_seed[0, i, e-1],robot_seed[0,i,e] ])
                batch_inputs[counter] = np.concatenate([self.type_one_hot[cell_input].flatten(), 
                                                        self.loc_one_hot[i-1], self.loc_one_hot[e-1], 
                                                        ])
                counter += 1
        return batch_inputs.flatten() # original: no flatten
    
    def get_obs_catch(self, robot_pos_final, package_pos_final):
        
        robot_com_pos = np.mean(robot_pos_final, axis=1)
        package_com_pos = np.mean(package_pos_final, axis=1)

        obs = np.array([
            package_com_pos[0]-robot_com_pos[0], package_com_pos[1]-robot_com_pos[1],
        ])
        return obs

    def get_obs_mani(self, robot_pos_final, robot_vel_final, package_pos_final, package_vel_final):
        
        robot_com_pos = np.mean(robot_pos_final, axis=1)
        robot_com_vel = np.mean(robot_vel_final, axis=1)
        box_com_pos = np.mean(package_pos_final, axis=1)
        box_com_vel = np.mean(package_vel_final, axis=1)

        obs = np.array([
            robot_com_vel[0], robot_com_vel[1],
            box_com_pos[0]-robot_com_pos[0], box_com_pos[1]-robot_com_pos[1],
            box_com_vel[0], box_com_vel[1]
        ])
        return obs

    def get_obs_topple(self, robot_pos_final, beam_pos_final):

        beam_com_pos_final = np.mean(beam_pos_final, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        diff = beam_com_pos_final - robot_com_pos_final
        return np.array([diff[0], diff[1]])

    def get_other_obs(self):
        # Get task-related state
        if self.env_id in ['Walker-v0', 'Climber-v0','Climber-v1']:
            return self.env.get_vel_com_obs("robot")
        elif self.env_id in ['BridgeWalker-v0']: 
            return np.concatenate((self.env.get_vel_com_obs("robot"), self.env.get_ort_obs("robot"))) 
        elif self.env_id in ['CaveCrawler-v0']:
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            self.env.get_floor_obs("robot", ["terrain"], self.env.sight_dist),
            self.env.get_ceil_obs("robot", ["terrain"], self.env.sight_dist),
            ))
        elif self.env_id in ['Balancer-v0','Balancer-v1',"Flipper-v0"]:
            return np.array([self.env.object_orientation_at_time(self.env.get_time(), "robot")]) 
        elif "mizer" in self.env_id:
            return np.array([0.0, 0.0])
        elif self.env_id in ['Climber-v2']:
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            self.env.get_ort_obs("robot"),
            self.env.get_ceil_obs("robot", ["pipe"], self.env.sight_dist),
            ))
        elif self.env_id in ['Jumper-v0']:
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            self.env.get_floor_obs("robot", ["ground"], self.env.sight_dist),
            ))
        elif self.env_id in ['Carrier-v0','Carrier-v1','Pusher-v0','Pusher-v1','Thrower-v0','Lifter-v0']:
            # collect post step information
            robot_pos_final = self.env.object_pos_at_time(self.env.get_time(), "robot")
            robot_vel_final = self.env.object_vel_at_time(self.env.get_time(), "robot")
            package_pos_final = self.env.object_pos_at_time(self.env.get_time(), "package")
            package_vel_final = self.env.object_vel_at_time(self.env.get_time(), "package")
            # observation
            obs = self.get_obs_mani(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)
            
            if self.env_id == 'Lifter-v0':
                obs = np.concatenate((obs,self.env.get_ort_obs("package"),))
            return obs  
        elif self.env_id in ['Catcher-v0']:
            # collect post step information
            robot_pos_final = self.env.object_pos_at_time(self.env.get_time(), "robot")
            package_pos_final = self.env.object_pos_at_time(self.env.get_time(), "package")

            # observation
            obs = self.get_obs_catch(robot_pos_final, package_pos_final)
            obs = np.concatenate((
                obs,
                self.env.get_vel_com_obs("robot"),
                self.env.get_vel_com_obs("package"),
                self.env.get_ort_obs("package"),
            ))
            return obs       
        elif self.env_id in ['BeamToppler-v0', 'BeamSlider-v0']:
            # collect post step information
            robot_pos_final = self.env.object_pos_at_time(self.env.get_time(), "robot")
            beam_pos_final = self.env.object_pos_at_time(self.env.get_time(), "beam")

            # observation
            obs = self.get_obs_topple(robot_pos_final, beam_pos_final)
            obs = np.concatenate((
                obs,
                self.env.get_vel_com_obs("robot"),
                self.env.get_vel_com_obs("beam"),
                self.env.get_ort_obs("beam"),
            ))
            return obs
        elif self.env_id in ['UpStepper-v0','DownStepper-v0','ObstacleTraverser-v0','ObstacleTraverser-v1']:
            robot_ort_final = self.env.object_orientation_at_time(self.env.get_time(), "robot")
            # observation
            obs = np.concatenate((
                self.env.get_vel_com_obs("robot"),
                np.array([robot_ort_final]),
                self.env.get_floor_obs("robot", ["ground"], self.env.sight_dist),
                ))
            return obs     
        elif self.env_id in ['Hurdler-v0']:
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            self.env.get_ort_obs("robot"),
            self.env.get_floor_obs("robot", ["ground"], self.env.sight_dist),
            ))
        elif self.env_id in ['PlatformJumper-v0']:
            robot_ort = self.env.object_orientation_at_time(self.env.get_time(), "robot")
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            np.array([robot_ort]),
            self.env.get_floor_obs("robot", self.env.terrain_list, self.env.sight_dist),
            ))

        elif self.env_id in ['GapJumper-v0','Traverser-v0']:    
            return np.concatenate((
            self.env.get_vel_com_obs("robot"),
            self.env.get_ort_obs("robot"),
            self.env.get_floor_obs("robot", self.env.terrain_list, self.env.sight_dist),
            ))

        else:
            raise ValueError("Env Error")  
        
    def transform_to_modular_obs(self, body):
        # This function is inspired by Mark Horton:
        # https://github.com/EvolutionGym/evogym/issues/6
        loc_idx = 0
        # index in obs vector where each corner will be stored. -1 if no value
        index_by_corner = np.zeros(tuple(body.shape) + (2, 2), dtype=int) - 1
        for y in range(body.shape[0]):
            for x in range(body.shape[1]):
                if body[y,x] != 0:
                    has_upper_neighbor = ((y-1) >= 0) and (body[y-1,x] != 0)
                    has_left_neighbor = ((x-1) >= 0) and (body[y,x-1] != 0)
                    has_right_neighbor = ((x+1) < body.shape[1]) and (body[y,x+1] != 0)
                    has_upper_right_neighbor = ((x+1) < body.shape[1]) and ((y-1) >= 0) and (body[y-1,x+1] != 0)

                    if has_upper_neighbor:
                        index_by_corner[y, x, 0, :] = index_by_corner[y - 1, x, 1, :]
                    if has_left_neighbor:
                        index_by_corner[y, x, :, 0] = index_by_corner[y, x - 1, :, 1]

                    if not (has_upper_neighbor or has_left_neighbor):
                        index_by_corner[y, x, 0, 0] = loc_idx
                        loc_idx += 1
                    if not has_upper_neighbor:
                        if has_right_neighbor and has_upper_right_neighbor:
                            index_by_corner[y, x, 0, 1] = index_by_corner[y-1, x+1, 1, 0]
                        else:
                            index_by_corner[y, x, 0, 1] = loc_idx
                            loc_idx += 1
                    if not has_left_neighbor:
                        index_by_corner[y, x, 1, 0] = loc_idx
                        loc_idx += 1

                    index_by_corner[y, x, 1, 1] = loc_idx
                    loc_idx += 1
        # Get index array
        index_array = self.modular_observation_array(index_by_corner)
        return index_array, loc_idx

    def modular_observation_array(self, index_by_corner):
        modular_ob_index = []
        for row in range(index_by_corner.shape[0]):
            for col in range(index_by_corner.shape[1]):
                block = index_by_corner[row][col]
                modular_ob_index.append(block.flatten())
        return np.array(modular_ob_index) # np.all(index_by_corner.reshape(-1, 4) == np.array(modular_ob_index))

    # Close
    def close(self):
        self.env.close()

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)
    