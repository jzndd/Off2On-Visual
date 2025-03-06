import d4rl
import gymnasium as gym
import numpy as np
import torch
import pickle

from rl_agents.utils import OfflineReplaybuffer

def store_d4rl_dataset(env_name: str, buffer: OfflineReplaybuffer):
    """
    Loads transitions from a D4RL dataset and stores them into an OfflineReplaybuffer.

    Args:
        env_name (str): Name of the D4RL environment.
        buffer (OfflineReplaybuffer): The replay buffer to store the dataset.
    """
    env = gym.make(env_name)
    dataset = env.get_dataset()

    # Extract transitions
    observations = dataset["observations"]
    next_observations = dataset["next_observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]

    # Store transitions into the buffer
    for i in range(len(observations)):
        obs = observations[i]
        act = actions[i]
        next_obs = next_observations[i]
        rew = rewards[i]
        done = terminals[i]

        buffer.store(obs, act, next_obs, rew, done)

    print(f"Stored {buffer.size} transitions in the buffer.")

    # file_save_path



# Example usage
env_name = "halfcheetah-medium-v2"  # Change this to your desired D4RL environment
buffer = OfflineReplaybuffer(capacity=1000000, obs_shape=(17,), act_dim=6)  # Update obs_shape and act_dim as needed
file_save_path = f"data/{env_name}"
store_d4rl_dataset(env_name, buffer)
