from transformer_ppo.run import run
import argparse
import numpy as np
from evogym import get_full_connectivity

# Robot to be controlled
# for Walker-v0
# robot_body = np.array([[3, 3, 4, 4, 4],
#                        [1, 4, 4, 4, 4],
#                        [3, 4, 4, 4, 4],
#                        [1, 4, 4, 4, 4],
#                        [3, 3, 3, 3, 4]])

# for Jumper-v0        
robot_body = np.array([[4,2,2,2,2,2,4],
                       [4,4,4,4,4,4,4],
                       [4,4,4,4,4,4,4],
                       [4,4,4,4,4,4,4],
                       [4,4,4,4,4,4,4],
                       [4,4,4,4,4,4,4],
                       [4,4,4,4,4,4,4],])
robot = [(robot_body, get_full_connectivity(robot_body))]

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='PyTorch args')
    parser.add_argument('--env', type=str, default="Jumper-v0",#'Walker-v0',
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
    args = parser.parse_args()

    # Run
    run(env_name=args.env, robots=robot, seed=args.seed, pop_size=args.threads_num,
        train_iters=args.train_iters, ac_type=args.ac_type, device_num=args.device_num)