# data/handle-pull-side-v2_96/expert_rb_with_reward_sparse_whole_traj.pkl

import pickle

import numpy as np
import torch

data = pickle.load(open('data/disassemble-v2_96/expert_rb_with_reward_sparse.pkl', 'rb'))
print(data.keys())

print(sum(data['rew']))