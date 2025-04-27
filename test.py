# data/handle-pull-side-v2_96/expert_rb_with_reward_sparse_whole_traj.pkl

import pickle

import numpy as np
import torch

data = pickle.load(open('data/coffee-pull-v2_96/expert_rb_with_reward_sparse_whole_traj.pkl', 'rb'))
print(data.keys())

print(len(data['obs']))