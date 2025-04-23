from collections import deque
from pathlib import Path
import d4rl
# import gymnasium as gym
import gym
import numpy as np
import torch
import pickle
import h5py

from rl_agent.utils import OfflineReplaybuffer

def store_d4rl_dataset(env_name: str, buffer: OfflineReplaybuffer, file_save_path: str):
    """
    Loads transitions from a D4RL dataset and stores them into an OfflineReplaybuffer.

    Args:
        env_name (str): Name of the D4RL environment.
        buffer (OfflineReplaybuffer): The replay buffer to store the dataset.
    """
    data_path = Path(f"/DATA/disk0/jzn/v-d4rl/vd4rl_data/{env_name}/84px")
    filenames = sorted(data_path.glob('*.hdf5'))
    lenghth = 0

    obs_buffer = deque([], maxlen=buffer.frame_stack)

    for filename in filenames:
        episodes = h5py.File(filename, 'r')
        dataset = {k: episodes[k][:] for k in episodes.keys()}
        # Extract transitions
        observations = dataset["observation"]
        # next_observations = dataset["next_observation"]
        actions = dataset["action"]
        rewards = dataset["reward"]
        step_type = dataset["step_type"] # 0 is begin, 1 is mid, 2 is end
        # terminals = dataset["terminal"]

        for i in range(len(observations)):
            if step_type[i] == 0:
                for _ in range(buffer.frame_stack):
                    obs_buffer.append(observations[i])  # 84 * 84 * 3
                done = 0.
                continue
            elif step_type[i] == 1:
                done = 0.
            else:
                done = 1.
            # obs = observations[i -1]
            stack_obs = np.concatenate(list(obs_buffer), axis=0) #  (3 * frame_stack) * 84 * 84
            act = actions[i]

            obs_buffer.append(observations[i]) 
            stack_next_obs = np.concatenate(list(obs_buffer), axis=0) #  (3 * frame_stack) * 84 * 84
            # next_obs = observations[i]
            rew = rewards[i]
            # done = terminals[i]
            buffer.store(stack_obs, act, stack_next_obs, rew, done)

        lenghth += len(observations)

    print("len of transitions", lenghth)
    print("buffer size", buffer.size)

    # Save the buffer to a file

    print(f"Stored {buffer.size} transitions in the buffer.")

    save_name = "expert_rb_with_reward.pkl"
    if frame_stack > 1:
        save_name = f"expert_rb_with_reward_stack{frame_stack}.pkl"

    buffer.save(f"{file_save_path}/{save_name}")
    # file_save_path

# Example usage
frame_stack = 3
env_name = "walker_walk"  # Change this to your desired D4RL environment
file_save_path = f"data/{env_name}_84"
import os
os.makedirs(file_save_path, exist_ok=True)
buffer = OfflineReplaybuffer(capacity=300000, action_shape=(6,), frame_stack=frame_stack)  # Update obs_shape and act_dim as needed
store_d4rl_dataset(env_name, buffer, file_save_path=file_save_path)

# env= gym.make("halfcheetah-medium-v2")
