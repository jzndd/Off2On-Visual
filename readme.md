### 按 policy 分布收集数据的一般流程


+ 收集纯 expert 数据以训 bc
```bash
# 请调整 img_size 参数 ！以及可能的 action_repeat 参数
# 其他参数说明：save_whole_traj （是否保存完整轨迹，训 world model 需要保存完整轨迹，训policy 不需要保存）
# use_sparse_reward 恒定为 True
# use_random 恒定为 False
python run_save_rb.py
```

+ train 一个 bc policy
```bash
python main2D.py only_bc=True img_size=128
```
训练好后将 outputs/...../bc.pth 手动放到 ckpt/..../bc.pth 中

+ 利用 bc policy 收集数据
```bash
python main2D.py save_data=True img_size=96
```  

启动save_data 时，会执行  

```python
trainer = Trainer(cfg, root_dir)
trainer.save_data()                 # save raw data (用于训练) 
trainer.save_eval_data()            # save test data (包括了 raw data 和 images 和 video， 去对应的 outputs/ 文件夹中 move 出来就行)
```

### vrl3 for metaworld
```python
python main2D_offpolicy.py common.devices=[6] ac_type=vrl3 task=door-lock-v2 actor_critic.training.batch_size=256

python main2D_offpolicy.py common.devices=[6] ac_type=vrl3 task=door-lock-v2 actor_critic.training.batch_size=256 is_sparse_reward=False

python main2D_offpolicy.py common.devices=[7] ac_type=vrl3 task=handle-pull-side-v2 actor_critic.training.batch_size=256

python main2D_offpolicy.py common.devices=[7] ac_type=vrl3 task=handle-pull-side-v2 actor_critic.training.batch_size=256 is_sparse_reward=False
```

### DrQv2+RLPD for metaworld

#### for metaworld
```python
python main2D_offpolicy_v2.py common.devices=[7] common.seed=10 ac_type=drqv2 task=door-lock-v2 actor_critic.training.batch_size=256 train_with_bc=False is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4

python main2D_offpolicy_v2.py common.devices=[7] common.seed=10 ac_type=drqv2 task=handle-pull-side-v2 actor_critic.training.batch_size=256 training.offline_steps=0 is_sparse_reward=False
```

#### for dmc
```python
python main2D_offpolicy_v2.py common.devices=[3] common.seed=10 ac_type=drqv2 task=walker_walk actor_critic.training.batch_size=256 train_with_bc=False is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4 img_size=84 evaluation.eval_times=5
```



### run script

```bash
python main.py task=walker2d-medium-v2 env=dmc common.devices=[0] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae agent.actor_critic_cfg.hidden_dim=64 common.seed=42 agent.actor_critic_cfg.online_lr=3e-4 training.bc_actor_warmup_steps=20000 training.bc_critic_warmup_steps=20000  

python main.py task=walker2d-medium-v2 env=dmc common.devices=[0] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae agent.actor_critic_cfg.hidden_dim=256 common.seed=42 agent.actor_critic_cfg.online_lr=3e-4 

python main.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=256

python main.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=1024 agent.ppo_cfg.mini_batch_size=128
```

### 2D
```bash
python main2D.py common.devices=[1] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=hammer-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=42 agent.actor_critic_cfg.hidden_dim=1024 is_sparse_reward=True

python main2D.py common.devices=[1] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=hammer-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=42 agent.actor_critic_cfg.hidden_dim=1024 agent.actor_critic_cfg.online_lr=3e-5 is_sparse_reward=True

python main2D.py common.devices=[0] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=button-press-topdown-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=42 agent.actor_critic_cfg.hidden_dim=1024 is_sparse_reward=True agent.actor_critic_cfg.online_lr=3e-5

python main2D.py common.devices=[0] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=button-press-topdown-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=42 agent.actor_critic_cfg.hidden_dim=1024 is_sparse_reward=True

# whole traj
python main2D.py common.devices=[0] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=button-press-topdown-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=100 agent.actor_critic_cfg.hidden_dim=1024 is_sparse_reward=True agent.actor_critic_cfg.online_lr=3e-5

python main2D.py common.devices=[1] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=hammer-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=100 agent.actor_critic_cfg.hidden_dim=1024 agent.actor_critic_cfg.online_lr=3e-5 is_sparse_reward=True

# save data
python main2D.py common.devices=[1] ac_type=ppo is_sparse_reward=True save_data=True task=hammer-v2 agent.actor_critic_cfg.hidden_dim=1024  only_bc=False 

# tradition
python main2D.py common.devices=[0] ac_type=ppo is_sparse_reward=True task=hammer-v2 agent.actor_critic_cfg.hidden_dim=1024  only_bc=False agent.ppo_cfg.adv_compute_mode=tradition actor_critic.training.batch_size=256 agent.ppo_cfg.mini_batch_size=256 seed=100

python main2D.py common.devices=[0] ac_type=ppo is_sparse_reward=True task=hammer-v2 agent.actor_critic_cfg.hidden_dim=1024  only_bc=False agent.ppo_cfg.adv_compute_mode=tradition actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=256 seed=200

# iql
python main2D.py common.devices=[1] ac_type=ppo is_sparse_reward=True task=hammer-v2 agent.actor_critic_cfg.hidden_dim=1024  only_bc=False agent.ppo_cfg.adv_compute_mode=iql actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=256 seed=100

python main2D.py common.devices=[1] ac_type=ppo is_sparse_reward=True task=hammer-v2 agent.actor_critic_cfg.hidden_dim=1024  only_bc=False agent.ppo_cfg.adv_compute_mode=iql actor_critic.training.batch_size=256 agent.ppo_cfg.mini_batch_size=256 seed=200

# tradition


```

### dmc control 
```bash
python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=iql agent.ppo_cfg.clip_param=0.2 common.devices=[7]

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=128 agent.ppo_cfg.adv_compute_mode=iql common.devices=[6] agent.ppo_cfg.clip_param=0.2

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=1024 agent.ppo_cfg.mini_batch_size=128 common.devices=[5] agent.ppo_cfg.adv_compute_mode=gae agent.ppo_cfg.clip_param=0.2

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=512 agent.ppo_cfg.mini_batch_size=128 common.devices=[4] agent.ppo_cfg.adv_compute_mode=gae agent.ppo_cfg.clip_param=0.2

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae common.devices=[6] agent.actor_critic_cfg.hidden_dim=64 common.seed=42

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=1024 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=iql common.devices=[6] agent.actor_critic_cfg.hidden_dim=64 common.seed=42

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae common.devices=[6] agent.actor_critic_cfg.hidden_dim=64 common.seed=42

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae common.devices=[0] agent.actor_critic_cfg.hidden_dim=64 common.seed=42 agent.actor_critic_cfg.online_lr=3e-4

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae common.devices=[1] agent.actor_critic_cfg.hidden_dim=256 common.seed=42
```












