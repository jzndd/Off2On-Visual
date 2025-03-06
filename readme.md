#### run script

'''python
python main.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=128

python main.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=256

python main.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=1024 agent.ppo_cfg.mini_batch_size=128
'''

'''python
# 2D
python main2D.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=128 task=button-press-topdown-v2

python main2D.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=256 task=button-press-topdown-v2

python main2D.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=1024 agent.ppo_cfg.mini_batch_size=128 task=button-press-topdown-v2
'''