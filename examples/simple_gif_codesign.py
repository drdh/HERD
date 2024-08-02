import os
root_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch
import gym
from utils.algo_utils import *
from transformer_ppo_codesign.envs import make_vec_envs
import transformer_ppo_codesign.helper as helper
import evogym.envs
import imageio
from PIL import Image
from pygifsicle import optimize
import re
from matplotlib import pyplot as plt
plt.ion() # matplotlib interactivate mode

def gen_gif(env_name='Walker-v0', seeds=[0,1,2,3], iterations=['last'], show_img=False, EXPERIMENT_PARENT_DIR=None):
    for seed in seeds:
        print(f"====================== seed {seed} =======================")
        for iteration in iterations:
            print(f"=========== iteration {iteration} =============")
            
            if EXPERIMENT_PARENT_DIR is None:
                EXPERIMENT_PARENT_DIR = os.path.join(root_dir, 'saved_data',f'{env_name}', f'codesign_{env_name}_seed_{seed}')

            print(f"{EXPERIMENT_PARENT_DIR}, iter={iteration}")

            if iteration == 'last':
                # sort_func = lambda x: int(re.findall(r"\d+\.?\d*", x)[0])
                sort_func = lambda x: int(re.findall(r"\d+", x)[0])
                iteration = sort_func(sorted(os.listdir(os.path.join(EXPERIMENT_PARENT_DIR, "controllers")), key=sort_func)[-1])
                
                # structure_filename = sorted(os.listdir(os.path.join(EXPERIMENT_PARENT_DIR, "structures")), key=sort_func)[-1]
                # controller_filename = sorted(os.listdir(os.path.join(EXPERIMENT_PARENT_DIR, "controllers")), key=sort_func)[-1]
                # save_path_structure = os.path.join(EXPERIMENT_PARENT_DIR, "structures", structure_filename)
                # save_path_controller = os.path.join(EXPERIMENT_PARENT_DIR, "controllers", controller_filename)
                # print(structure_filename, controller_filename)

            try:
                save_path_structure = os.path.join(EXPERIMENT_PARENT_DIR, "structures", f"iter_{iteration}.npz")
                save_path_controller = os.path.join(EXPERIMENT_PARENT_DIR, "controllers", f"iter_{iteration}.pt")
        
                gif_path = f'{EXPERIMENT_PARENT_DIR}/{env_name}_seed_{seed}_iter_{iteration}.gif'

                structure_data = np.load(save_path_structure)
                structure = []
                for key, value in structure_data.items():
                    structure.append(value)
                structure = tuple(structure)
                robot = Structure(*structure, 0) 

                env = make_vec_envs(env_name, [robot], seed=1, gamma=None, device='cpu', ret=False, ob=True)
                                
                uni_agent, obs_rms = torch.load(save_path_controller, map_location='cpu')
                uni_agent.eval()
                uni_agent.to('cpu')

                vec_norm = helper.get_vec_normalize(env)
                if vec_norm is not None:
                    vec_norm.eval()
                    vec_norm.ob_rms = obs_rms

                obs = env.reset()
                imgs = []
                eval_episode_rewards = []
                # Rollout
                t = 0
                total_reward = 0
                while True:
                    with torch.no_grad():
                        val, action, logp,  = uni_agent.uni_act(obs, mean_action=True)
                    obs, reward, done, infos = env.step(action)
                    t += 1
                    total_reward += reward.item()
                    img = env.render(mode='img')  
                    imgs.append(img)

                    if show_img:
                        plt.imshow(img)
                        plt.title(f"t={t}, reward={reward.item():.4f}, total={total_reward:.4f}")
                        # plt.show()
                        plt.pause(0.01)
                        plt.clf() # clear the current figure

                    for info in infos:
                        if 'episode' in info.keys():
                            eval_episode_rewards.append(info['episode']['r'])
                    # Done
                    if len(eval_episode_rewards)==1:
                        break
                env.close()
                print("Evalution done!")
                print(eval_episode_rewards)

                imageio.mimsave(gif_path, imgs, duration=(1/50.0))
                print("GIF save to : ", gif_path)
                
            except Exception as e:
                print(e)

if __name__ == '__main__':

    env_name='Walker-v0', # easy
    # env_name='BridgeWalker-v0', # easy
    # env_name='BidirectionalWalker-v0', # medium
    # env_name='Carrier-v0', # easy
    # env_name='Carrier-v1', # hard
    # env_name='Pusher-v0', # easy
    # env_name='Pusher-v1', # medium
    # env_name='Thrower-v0', # medium
    # env_name='Catcher-v0', # hard
    # env_name='BeamToppler-v0', # easy
    # env_name='BeamSlider-v0', # hard
    # env_name='Lifter-v0', # hard
    # env_name='Climber-v0', # medium
    # env_name='Climber-v1', # medium
    # env_name='Climber-v2', # hard
    # env_name='UpStepper-v0', # medium
    # env_name='DownStepper-v0', # easy
    # env_name='ObstacleTraverser-v0', # medium
    # env_name='ObstacleTraverser-v1', # hard
    # env_name='Hurdler-v0', # hard
    # env_name='PlatformJumper-v0', # hard
    # env_name='GapJumper-v0', # hard
    # env_name='Traverser-v0', # hard
    # env_name='CaveCrawler-v0', # medium
    # env_name='AreaMaximizer-v0', # easy
    # env_name='AreaMinimizer-v0', # medium
    # env_name='WingspanMazimizer-v0', # easy
    # env_name='HeightMaximizer-v0', # medium
    # env_name='Flipper-v0', # easy
    # env_name='Jumper-v0', # easy
    # env_name='Balancer-v0', # easy
    # env_name='Balancer-v1' # medium

    seeds = [0, 1, 2, 3] 
    # seeds = [4, 5, 6, 7]
    # seeds = [10, 11, 12, 13]
    # seeds = [14, 15, 16, 17]
    
    iterations = ['last']
    show_img = False

    gen_gif(env_name, seeds, iterations, show_img)




   