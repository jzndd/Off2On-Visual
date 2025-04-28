from rl_agent.utils import EfficientReplayBuffer, EfficientReplayBufferV2

import numpy as np
import time

expertrb = EfficientReplayBufferV2(1000000, (9,84,84), (6,), 3)
expertrb.load("/data/jzn/workspace/ppo_alg/Off2On-Visual/data/walker_walk_84/efficient_rb_with_reward_stack3_backup.pkl")

for _ in range(10):
    time_start = time.time()
    for _ in range(10):
        batch = expertrb.sample(256)
    time_end = time.time()
    elapsed_time = time_end - time_start
    print(f"Elapsed time for 10 iterations: {elapsed_time:.2f} seconds")

# expertrb = EfficientReplayBuffer(1000000, (9,84,84), (6,), 3)
# expertrb.load("/data/jzn/workspace/ppo_alg/Off2On-Visual/data/walker_walk_84/efficient_rb_with_reward_stack3.pkl")

# for _ in range(10):
#     time_start = time.time()
#     for _ in range(10):
#         batch = expertrb.sample(256)
#     time_end = time.time()
#     elapsed_time = time_end - time_start
#     print(f"Elapsed time for 10 iterations: {elapsed_time:.2f} seconds")