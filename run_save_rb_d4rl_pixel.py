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
                continue
            elif step_type[i] == 1:
                done = 0.
            else:
                done = 1.
            obs = observations[i -1]
            act = actions[i]
            next_obs = observations[i]
            rew = rewards[i]
            # done = terminals[i]
            buffer.store(obs, act, next_obs, rew, done)

        lenghth += len(observations)

    print("len of transitions", lenghth)
    print("buffer size", buffer.size)

    # Save the buffer to a file

    print(f"Stored {buffer.size} transitions in the buffer.")
    buffer.save(f"{file_save_path}/expert_rb_with_reward_state.pkl")
    # file_save_path

# Example usage
env_name = "walker-walk"  # Change this to your desired D4RL environment
file_save_path = f"data/{env_name}-pixels"
import os
os.makedirs(file_save_path, exist_ok=True)
buffer = OfflineReplaybuffer(capacity=300000, action_shape=(6,))  # Update obs_shape and act_dim as needed
store_d4rl_dataset(env_name, buffer, file_save_path=file_save_path)

# env= gym.make("halfcheetah-medium-v2")
