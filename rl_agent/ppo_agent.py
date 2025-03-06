from collections import namedtuple
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

from omegaconf import ListConfig
import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

# from coroutines.env_loop import make_env_loop
# from envs import TorchEnv, WorldModelEnv
# from utils import init_lstm, LossAndLogs

import os
import sys

from .utils import *
from torch.distributions import Normal
from .baseagent import *
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler  

from .net import *

from copy import deepcopy

@dataclass
class PPOConfig:
    clip_param: float = 0.2
    K_epochs: int = 6
    mini_batch_size: int = 256
    max_grad_norm: float = 0.5 

class PPOAgent2D(BaseAgent):
    def __init__(
        self,
        cfg: ActorCriticConfig,
        ppo_cfg: PPOConfig,
        use_old_policy: bool = True,
        adv_compute_mode: str = "tradition", # optional:"tradition", "gae", "iql"
    ):
        super().__init__(cfg)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## initialize ppo hyperparameters
        self.clip_param =  ppo_cfg.clip_param
        self.K_epochs =  ppo_cfg.K_epochs
        self.mini_batch_size =  ppo_cfg.mini_batch_size
        self.max_grad_norm =  ppo_cfg.max_grad_norm

        ## network initialization
        self.encoder = ActorCriticEncoder()
        self.actor = Actorlog(cfg, self.encoder.repr_dim)
        # self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        # self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=1e-4)

        self.adv_compute_mode = adv_compute_mode

        if self.adv_compute_mode in ['tradition', 'gae']:
            # use GAE to estimate the advantage
            self.value = ValueMLP(cfg, self.encoder.repr_dim)
            # self.value_opt = optim.Adam(self.value.parameters(), lr=1e-4)
        elif self.adv_compute_mode == 'iql':
            self.critic = IQLCritic(cfg, self.encoder.repr_dim) 
        else:
            raise NotImplementedError

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.actor.parameters())
                                    + list(self.value.parameters()), lr=1e-4)

        # self.use_old_policy = use_old_policy
        # if self.use_old_policy:
        #     self.actor_old = deepcopy(self.actor)
        #     self.actor_old.requires_grad_(False)
        
        # self.lr_actor_scheduler = optim.lr_scheduler.StepLR(self.actor_opt, step_size=1000, gamma=0.9)

        # self.bc_loss = nn.MSELoss()

        self.ac_type = "ppo"
        
    def predict_act(self, obs: torch.Tensor, eval_mode=False) -> ActorCriticOutput:
        assert obs.ndim == 4  # Ensure observation shape is correct
        h = self.encoder(obs).flatten(start_dim=1)
        if self.adv_compute_mode == 'tradition' and not eval_mode:
            # When tradition, return both action and state_value
            return *self.actor.get_action(h, eval_mode=eval_mode), self.value(h)
        return self.actor.get_action(h, eval_mode=eval_mode)
    
    def bc_update(self, rb):

        c = self.loss_cfg

        if isinstance(rb, OfflineReplaybuffer):

            # update policy
            bacth = rb.sample(mini_batch_size=256)
            obs, act, reward, done, returns, obs_ = to_torch(bacth, self.device)
            # obs, act, reward, done, returns, obs_ = rb.sample(mini_batch_size=256)

            x = self.encoder(obs).flatten(start_dim=1)
            # x = x.flatten(start_dim=1)
            dist = self.actor.get_dist(x)
            log_prob = dist.log_prob(act)
            bc_loss = -log_prob.sum(dim=-1).mean() 
            bc_loss = c.weight_bc_loss * bc_loss

            # self.actor_opt.zero_grad()
            # self.encoder_opt.zero_grad()
            # bc_loss.backward()
            # self.actor_opt.step()
            # self.encoder_opt.step()

            # update critic network and freeze the encoder network
            if self.adv_compute_mode in ['tradition', 'gae']:
                val = self.value(x.detach())
                value_loss = F.mse_loss(val, returns)
                q_loss = torch.tensor(0.0)
                # update value loss
                # self.critic_opt.zero_grad()
                # value_loss.backward()
                # self.critic_opt.step()
                # value_loss = value_loss.item()
            else:
                self.critic: IQLCritic
                x_ = self.encoder(obs_).flatten(start_dim=1)
                q_loss, value_loss = self.critic.update(x.detach(), act, 1-done, returns, x_)
            # bc_loss = c.weight_bc_loss * self.bc_loss(pred_act, act)

        loss = bc_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"bc_loss":bc_loss.item(), 'value_loss':value_loss.item(), "q_loss":q_loss.item()}   
    
    def bc_transfer_ac(self):
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=2e-6) # lower the learning rate
        self.loss_cfg.weight_bc_loss = 0.1
        self.fix_encoder = True           # When online, fix encoder
        self.optimizer = optim.Adam(list(self.actor.parameters())
                                    + list(self.value.parameters()), lr=2e-6)
    
    def update(self, rb: OnlineReplayBuffer, batch=None):

        c = self.loss_cfg
    
        batch = rb.sample(rb.size)
        obss, acts, rews, next_obss, dones, old_log_probs, old_state_values = to_torch(batch, self.device)

        old_log_probs = old_log_probs.squeeze(-1)
        old_state_values = old_state_values.squeeze(-1) if old_state_values is not None else None

        returns = compute_returns(rews, 0.99, dones)
        returns = returns.to(self.device)
        returns = returns.squeeze(-1)
        
        if self.fix_encoder:
            with torch.no_grad():
                x = self.encoder(obss).reshape(obss.shape[0], -1)
        else:
            x = self.encoder(obss).reshape(obss.shape[0], -1)

        if self.adv_compute_mode == 'tradition':  
            # rerference: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/728cce83d7ab628fe2634eabcdf3239997eb81dd/PPO.py#L221
            advantages = returns.detach() - old_state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        # # returns = compute_lambda_returns(rews, dones, truncs, val, c.gamma, c.lambda_)
        # advantages = (returns - val).detach()

        # value_loss = F.mse_loss(val, returns)

        # self.critic_opt.zero_grad()
        # value_loss.backward()
        # self.critic_opt.step()

        batch_size = rb.size

        for i in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, False):
                # if not self.use_old_policy:
                dist = self.actor.get_dist(x[index].detach())
                log_prob = dist.log_prob(acts[index]).sum(dim=1)
                # else:
                #     # old
                #     old_dist = self.actor_old.get_dist(x[index].detach())
                #     old_action = old_dist.rsample()
                #     old_log_prob = old_dist.log_prob(old_action).sum(dim=-1, keepdim=True)
                #     # new
                #     dist = self.actor.get_dist(x[index].detach())
                #     log_probs = dist.log_prob(old_action).sum(dim=-1, keepdim=True)

                #     with torch.no_grad():
                #         self.critic: IQLCritic
                #         advantages = self.critic.get_advantage(x[index], old_action) 
                    

                    # log_probs = dist.log_prob(acts[index]).sum(dim=1)

                # if self.adv_compute_mode == 'tradition':
                #     advantages = advantages[index]

                if self.adv_compute_mode == "tradition":
                    state_value = self.value(x[index].detach())
                    value_loss = F.mse_loss(state_value.squeeze(), returns[index])
        
                entropy_loss = -c.weight_entropy_loss * dist.entropy().mean()
                surrogate = -advantages[index] * torch.exp(log_prob - old_log_probs[index].detach())
                surrogate_clipped = -advantages[index] * torch.clamp(torch.exp(log_prob - old_log_probs[index].detach()),
                                                       1.0 - self.clip_param, 1.0 + self.clip_param)
                # if batch is not None:
                #     expert_obs, expert_act, _, _, _ = batch.sample(mini_batch_size=256)
                #     expert_act, expert_obs = expert_act.to(self.device), expert_obs.to(self.device)
                #     expert_x = self.encoder(expert_obs)
                #     expert_x = expert_x.flatten(start_dim=1)
                #     dist = self.actor.get_dist(expert_x)
                #     log_prob = dist.log_prob(expert_act)
                #     bc_loss = -log_prob.sum(dim=-1).mean()
                #     bc_loss = c.weight_bc_loss * bc_loss
                # else:
                #     bc_loss = 0.0
                
                policy_loss = torch.max(surrogate, surrogate_clipped).mean() + entropy_loss

                loss = policy_loss + 0.5 * value_loss
        
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # if batch is not None:
        #     bc_loss_dict = self.bc_update(batch)
        #     bc_loss = bc_loss_dict['bc_loss']
        #     value_loss = bc_loss_dict['value_loss'] 
        # else:
        #     bc_loss = 0.0
        #     value_loss = 0.0

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss, "entropy_loss": entropy_loss.item(),
                "surrogate": surrogate.mean().item(), "surrogate_clipped": surrogate_clipped.mean().item()} 
    
    # def set_old_policy(self):
    #     self.actor_old.load_state_dict(self.actor.state_dict())

class PPOAgent(BaseAgent):
    def __init__(
        self,
        cfg: ActorCriticConfig,
        ppo_cfg: PPOConfig,
        use_old_policy: bool = True,
        adv_compute_mode: str = "tradition", # optional:"tradition", "gae", "iql"
    ):
        super().__init__(cfg)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## initialize ppo hyperparameters
        self.clip_param =  ppo_cfg.clip_param
        self.K_epochs =  ppo_cfg.K_epochs
        self.mini_batch_size =  ppo_cfg.mini_batch_size
        self.max_grad_norm =  ppo_cfg.max_grad_norm
        state_dim = 39
        ## network initialization

        self.actor = Actorlog(cfg, state_dim, use_trunk=False)

        self.adv_compute_mode = adv_compute_mode

        if self.adv_compute_mode in ['tradition', 'gae']:
            # use GAE to estimate the advantage
            self.value = ValueMLP(cfg, state_dim, use_trunk=False)
            # self.value_opt = optim.Adam(self.value.parameters(), lr=1e-4)
        elif self.adv_compute_mode == 'iql':
            self.critic = IQLCritic(cfg, state_dim, use_trunk=False) 
        else:
            raise NotImplementedError

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.value.parameters()), lr=1e-4)

        # self.use_old_policy = use_old_policy
        # if self.use_old_policy:
        #     self.actor_old = deepcopy(self.actor)
        #     self.actor_old.requires_grad_(False)
        
        # self.lr_actor_scheduler = optim.lr_scheduler.StepLR(self.actor_opt, step_size=1000, gamma=0.9)

        # self.bc_loss = nn.MSELoss()

        self.ac_type = "ppo"
        
    def predict_act(self, obs: torch.Tensor, eval_mode=False) -> ActorCriticOutput:
        assert obs.ndim == 2  # Ensure observation shape is correct
        if self.adv_compute_mode == 'tradition' and not eval_mode:
            # When tradition, return both action and state_value
            return *self.actor.get_action(obs, eval_mode=eval_mode), self.value(obs)
        return self.actor.get_action(obs, eval_mode=eval_mode)
    
    def bc_update(self, rb):

        c = self.loss_cfg

        if isinstance(rb, OfflineReplaybuffer):

            # update policy
            bacth = rb.sample(mini_batch_size=256)
            obs, act, reward, done, returns, obs_ = to_torch(bacth, self.device)
            # obs, act, reward, done, returns, obs_ = rb.sample(mini_batch_size=256)

            # x = x.flatten(start_dim=1)
            dist = self.actor.get_dist(obs)
            log_prob = dist.log_prob(act)
            bc_loss = -log_prob.sum(dim=-1).mean() 
            bc_loss = c.weight_bc_loss * bc_loss

            # self.actor_opt.zero_grad()
            # self.encoder_opt.zero_grad()
            # bc_loss.backward()
            # self.actor_opt.step()
            # self.encoder_opt.step()

            # update critic network and freeze the encoder network
            if self.adv_compute_mode in ['tradition', 'gae']:
                val = self.value(obs)
                value_loss = F.mse_loss(val, returns)
                q_loss = torch.tensor(0.0)
                # update value loss
                # self.critic_opt.zero_grad()
                # value_loss.backward()
                # self.critic_opt.step()
                # value_loss = value_loss.item()
            else:
                self.critic: IQLCritic
                q_loss, value_loss = self.critic.update(obs, act, 1-done, returns, obs_)
            # bc_loss = c.weight_bc_loss * self.bc_loss(pred_act, act)

        loss = bc_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"bc_loss":bc_loss.item(), 'value_loss':value_loss.item(), "q_loss":q_loss.item()}   
    
    def bc_transfer_ac(self):
        self.optimizer = optim.Adam(list(self.actor.parameters())
                                    + list(self.value.parameters()), lr=2e-6)
    
    def update(self, rb: OnlineReplayBuffer, batch=None):

        c = self.loss_cfg
    
        batch = rb.sample(rb.size)
        obss, acts, rews, next_obss, dones, old_log_probs, old_state_values = to_torch(batch, self.device)

        old_log_probs = old_log_probs.squeeze(-1)
        old_state_values = old_state_values.squeeze(-1) if old_state_values is not None else None

        returns = compute_returns(rews, 0.99, dones)
        returns = returns.to(self.device)
        returns = returns.squeeze(-1)

        if self.adv_compute_mode == 'tradition':  
            # rerference: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/728cce83d7ab628fe2634eabcdf3239997eb81dd/PPO.py#L221
            advantages = returns.detach() - old_state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        batch_size = rb.size

        for i in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, False):
                # if not self.use_old_policy:
                dist = self.actor.get_dist(obss[index])
                log_prob = dist.log_prob(acts[index]).sum(dim=1)
                # else:
                #     # old
                #     old_dist = self.actor_old.get_dist(x[index].detach())
                #     old_action = old_dist.rsample()
                #     old_log_prob = old_dist.log_prob(old_action).sum(dim=-1, keepdim=True)
                #     # new
                #     dist = self.actor.get_dist(x[index].detach())
                #     log_probs = dist.log_prob(old_action).sum(dim=-1, keepdim=True)

                #     with torch.no_grad():
                #         self.critic: IQLCritic
                #         advantages = self.critic.get_advantage(x[index], old_action) 
                    

                    # log_probs = dist.log_prob(acts[index]).sum(dim=1)

                # if self.adv_compute_mode == 'tradition':
                #     advantages = advantages[index]

                if self.adv_compute_mode == "tradition":
                    state_value = self.value(obss[index].detach())
                    value_loss = F.mse_loss(state_value.squeeze(), returns[index])
        
                entropy_loss = -c.weight_entropy_loss * dist.entropy().mean()
                surrogate = -advantages[index] * torch.exp(log_prob - old_log_probs[index].detach())
                surrogate_clipped = -advantages[index] * torch.clamp(torch.exp(log_prob - old_log_probs[index].detach()),
                                                       1.0 - self.clip_param, 1.0 + self.clip_param)
                # if batch is not None:
                #     expert_obs, expert_act, _, _, _ = batch.sample(mini_batch_size=256)
                #     expert_act, expert_obs = expert_act.to(self.device), expert_obs.to(self.device)
                #     expert_x = self.encoder(expert_obs)
                #     expert_x = expert_x.flatten(start_dim=1)
                #     dist = self.actor.get_dist(expert_x)
                #     log_prob = dist.log_prob(expert_act)
                #     bc_loss = -log_prob.sum(dim=-1).mean()
                #     bc_loss = c.weight_bc_loss * bc_loss
                # else:
                #     bc_loss = 0.0
                
                policy_loss = torch.max(surrogate, surrogate_clipped).mean() + entropy_loss

                loss = policy_loss + 0.5 * value_loss
        
                self.optimizer.zero_grad()
                loss.backward()
                params_list = list(self.actor.parameters()) + list(self.value.parameters())
                nn.utils.clip_grad_norm_(params_list, self.max_grad_norm)
                self.optimizer.step()
        
        # if batch is not None:
        #     bc_loss_dict = self.bc_update(batch)
        #     bc_loss = bc_loss_dict['bc_loss']
        #     value_loss = bc_loss_dict['value_loss'] 
        # else:
        #     bc_loss = 0.0
        #     value_loss = 0.0

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss, "entropy_loss": entropy_loss.item(),
                "surrogate": surrogate.mean().item(), "surrogate_clipped": surrogate_clipped.mean().item()} 