from transformer_ppo_codesign.run import run
import argparse
import numpy as np
from evogym import get_full_connectivity


robot_body = np.array([
    [1, 4, 0, 2, 1, 2, 3, 1],
    [0, 3, 1, 3, 4, 2, 4, 0],
    [3, 3, 0, 4, 0, 0, 1, 2],
    [3, 4, 0, 2, 2, 2, 1, 0],
    [4, 4, 3, 1, 3, 4, 4, 4],
    [2, 1, 3, 2, 3, 3, 0, 4],
    [0, 0, 1, 4, 2, 2, 3, 4],
    [2, 2, 3, 4, 0, 0, 4, 3]
])

wh = 5 # robot_body.shape[0] # width & height
robot_body = robot_body[:wh, :wh]

robot = [(robot_body, get_full_connectivity(robot_body))]

load_iter =  None  

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='PyTorch args')
    parser.add_argument('--env', type=str, 
                        # default='Walker-v0', # easy
                        # default='BridgeWalker-v0', # easy
                        # default='BidirectionalWalker-v0', # medium
                        # default='Carrier-v0', # easy
                        # default='Carrier-v1', # hard
                        # default='Pusher-v0', # easy
                        # default='Pusher-v1', # medium
                        # default='Thrower-v0', # medium
                        # default='Catcher-v0', # hard
                        # default='BeamToppler-v0', # easy
                        # default='BeamSlider-v0', # hard
                        # default='Lifter-v0', # hard
                        # default='Climber-v0', # medium
                        # default='Climber-v1', # medium
                        # default='Climber-v2', # hard
                        # default='UpStepper-v0', # medium
                        # default='DownStepper-v0', # easy
                        default='ObstacleTraverser-v0', # medium
                        # default='ObstacleTraverser-v1', # hard
                        # default='Hurdler-v0', # hard
                        # default='PlatformJumper-v0', # hard
                        # default='GapJumper-v0', # hard
                        # default='Traverser-v0', # hard
                        # default='CaveCrawler-v0', # medium
                        # default='AreaMaximizer-v0', # easy
                        # default='AreaMinimizer-v0', # medium
                        # default='WingspanMazimizer-v0', # easy
                        # default='HeightMaximizer-v0', # medium
                        # default='Flipper-v0', # easy
                        # default='Jumper-v0', # easy
                        # default='Balancer-v0', # easy
                        # default='Balancer-v1' # medium
                        help='tasks')
    parser.add_argument('--seed', type=int, default=101,
                        help='random seed')
    parser.add_argument('--ac_type', type=str, default="transformer",
                        help='(transformer, fc)')
    parser.add_argument('--device_num', type=int, default=1,
                        help='gpu device id')
    parser.add_argument('--threads_num', type=int, default=4,
                        help='number of threads')
    parser.add_argument('--train_iters', type=int, default=1000,
                        help='training iterations')  
    parser.add_argument('--save_replay', type=int, default=1,
                        help="save replay or not")                                      
    args = parser.parse_args()

    # Run
    run(env_name=args.env, robots=robot, seed=args.seed, pop_size=args.threads_num,
        train_iters=args.train_iters, ac_type=args.ac_type, device_num=args.device_num, wh=wh, 
        load_iter=load_iter, save_replay=args.save_replay)