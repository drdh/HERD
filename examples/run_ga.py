import random
import numpy as np

from ga.run import run_ga

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    run_ga(
        pop_size = 3, #25, 3,
        structure_shape = (5,5),
        experiment_name = "ga",
        max_evaluations = 60, # 60, 600
        train_iters = 50, # 50, 1000
        num_cores = 3,
        seed = seed,
        # env_name='Walker-v0', # easy
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
        env_name='Jumper-v0', # easy
        # env_name='Balancer-v0', # easy
        # env_name='Balancer-v1' # medium
    )