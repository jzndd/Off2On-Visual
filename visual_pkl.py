import os
import pickle as pkl
import numpy as np

file_path = "/data/home/jzn/workspace/Off2On-Visual/data/coffee-pull-v2_64/expert_rb_with_reward_sparse.pkl"

data = pkl.load(open(file_path, "rb"))
print(data['done'].sum())



# handle-pull-v2 50 ; 
# soccer-v2 50 ;
# door-lock-v2 50 ;
# coffee-pull-v2 20 ;
# lever-pull-v2 20 ;
# hammer-v2 20 ;
