import d4rl
# import gymnasium as gym
import gym
import numpy as np
import torch
import pickle

from rl_agent.utils import OfflineReplaybuffer

def store_d4rl_dataset(env_name: str, buffer: OfflineReplaybuffer, file_save_path: str):
    """
    Loads transitions from a D4RL dataset and stores them into an OfflineReplaybuffer.

    Args:
        env_name (str): Name of the D4RL environment.
        buffer (OfflineReplaybuffer): The replay buffer to store the dataset.
    """
    env = gym.make("halfcheetah-medium-v2")
    dataset = env.get_dataset()

    # Extract transitions
    observations = dataset["observations"]
    next_observations = dataset["next_observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]

    print("len of transitions", len(observations))

    # Store transitions into the buffer
    for i in range(len(observations)):
        obs = observations[i]
        act = actions[i]
        next_obs = next_observations[i]
        rew = rewards[i]
        done = terminals[i]

        buffer.store(obs, act, next_obs, rew, done)

    print(f"Stored {buffer.size} transitions in the buffer.")
    buffer.save(f"{file_save_path}/expert_rb_with_reward_state.pkl")
    # file_save_path

# Example usage
env_name = "halfcheetah-medium-v2"  # Change this to your desired D4RL environment
file_save_path = f"data/{env_name}"
import os
os.makedirs(file_save_path, exist_ok=True)
buffer = OfflineReplaybuffer(capacity=110000, obs_shape=(17,), action_shape=(6,))  # Update obs_shape and act_dim as needed
store_d4rl_dataset(env_name, buffer, file_save_path=file_save_path)

# env= gym.make("halfcheetah-medium-v2")
