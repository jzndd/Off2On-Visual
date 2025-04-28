from collections import deque
from pathlib import Path
import d4rl
# import gymnasium as gym
import gym
import numpy as np
import torch
import pickle
import h5py

from rl_agent.utils import EfficientReplayBuffer

def store_d4rl_dataset(env_name: str, buffer: EfficientReplayBuffer, file_save_path: str):
    """
    Loads transitions from a D4RL dataset and stores them into an OfflineReplaybuffer.

    Args:
        env_name (str): Name of the D4RL environment.
        buffer (OfflineReplaybuffer): The replay buffer to store the dataset.
    """
    data_path = Path(f"/data/jzn/workspace/DiffusionRL/vd4rl/vd4rl_data/main/{env_name}/medium_expert/84px")
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
        discounts = dataset["discount"]
        step_type = dataset["step_type"] # 0 is begin, 1 is mid, 2 is end
        # terminals = dataset["terminal"]

        for i in range(len(observations)):
            if step_type[i] == 0:
                for _ in range(buffer.frame_stack):
                    obs_buffer.append(observations[i])  # 84 * 84 * 3
                rew = rewards[i]
                act = actions[i]
                stack_obs = np.concatenate(list(obs_buffer), axis=0)
                stack_obs = torch.as_tensor(stack_obs, device="cuda").div(255).mul(2).sub(1).contiguous().detach().cpu().numpy()
                dis = discounts[i]
                buffer.store(stack_obs, act, rew, dis, True)
                continue
            # elif step_type[i] == 1:
            #     done = 0.
            # else:
            #     done = 1.
            # obs = observations[i -1]
            # stack_obs = np.concatenate(list(obs_buffer), axis=0) #  (3 * frame_stack) * 84 * 84

            obs_buffer.append(observations[i]) 
            stack_obs = np.concatenate(list(obs_buffer), axis=0) #  (3 * frame_stack) * 84 * 84
            stack_obs = torch.as_tensor(stack_obs, device="cuda").div(255).mul(2).sub(1).contiguous().detach().cpu().numpy()
            rew = rewards[i]
            act = actions[i]
            dis = discounts[i]
            # done = terminals[i]
            buffer.store(stack_obs, act, rew, dis, False)

        lenghth += len(observations)

    print("len of transitions", lenghth)
    print("buffer size", buffer.size)

    # Save the buffer to a file

    print(f"Stored {buffer.size} transitions in the buffer.")

    save_name = "efficient_rb_with_reward_nofirst.pkl"
    if frame_stack > 1:
        save_name = f"efficient_rb_with_reward_stack{frame_stack}_nofirst.pkl"

    batch = buffer.sample(256)

    buffer.save(f"{file_save_path}/{save_name}")
    # file_save_path

# Example usage
frame_stack = 3
env_name = "walker_walk"  # Change this to your desired D4RL environment
file_save_path = f"data/{env_name}_84"
import os
os.makedirs(file_save_path, exist_ok=True)
buffer = EfficientReplayBuffer(capacity=300000, obs_shape=(9,84,84), act_shape=(6,), frame_stack=frame_stack)  # Update obs_shape and act_dim as needed
store_d4rl_dataset(env_name, buffer, file_save_path=file_save_path)

# env= gym.make("halfcheetah-medium-v2")
