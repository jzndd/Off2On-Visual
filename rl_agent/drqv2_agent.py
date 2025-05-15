# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from rl_agent.baseagent import ActorCriticConfig, BaseAgent
from rl_agent.utils import update_exponential_moving_average, merge_batches

from rl_agent.net import Actorlog, DoubleQMLP, DrQv2Actor, VRL3Actor, ActorCriticEncoder, EnsembleCritic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import os
import copy
from PIL import Image
import platform
from numbers import Number
from rl_agent import  utils

from dataclasses import dataclass

from rl_agent.baseagent import ActorCriticConfig

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class Identity(nn.Module):
    def __init__(self, input_placeholder=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 1
        self.repr_dim = obs_shape[0]

    def forward(self, obs):
        return obs
    
# DrQv2 Config
@dataclass
class DrQv2Config:
    critic_target_tau: float = 0.01
    num_expl_steps: int = 2000          # because there are no offline steps, so need num_expl_steps
    update_every_steps: int = 2
    stddev_clip: float = 0.3
    use_data_aug: bool = True
    encoder_lr_scale: float = 1
    stage2_update_encoder: bool = True # in this version, we will directly online
    stage2_std: float = 0.1
    stage3_update_encoder: bool = True
    std0: float = 1.0
    std1: float = 0.1
    std_n_decay: int = 100000          # 0.1m
    utd_ratio: float = 1
    mini_batch_size: int = 256
    num_critics: int = 2
    offline_data_ratio: float = 0.5
    bc_weight: float = 0.0  

# 先验证 Online
class DrQv2Agent(BaseAgent):
    def __init__(self, cfg: ActorCriticConfig, drqv2_cfg: DrQv2Config = None):
        
        super().__init__(cfg)

        # ====== stage 2, 3 ======
        if drqv2_cfg is None:
            drqv2_cfg = DrQv2Config()
        
        # ------------------------- set default values ------------------------
        action_dim = cfg.num_actions
        lr=1e-4
        self.ac_type = "vrl3"
        self.device = "cuda"
        self.stage = 2
        # ------------------------- set default values end ------------------------

        self.critic_target_tau = drqv2_cfg.critic_target_tau
        self.update_every_steps = drqv2_cfg.update_every_steps
        self.num_expl_steps = drqv2_cfg.num_expl_steps
        self.utd_ratio = drqv2_cfg.utd_ratio
        self.offline_data_ratio = drqv2_cfg.offline_data_ratio
        self.mini_batch_size = drqv2_cfg.mini_batch_size

        self.stage2_std = drqv2_cfg.stage2_std
        self.stage2_update_encoder = drqv2_cfg.stage2_update_encoder

        self.bc_weight = drqv2_cfg.bc_weight

        if drqv2_cfg.std1 > drqv2_cfg.std0:
            drqv2_cfg.std1 = drqv2_cfg.std0
        self.stddev_schedule = "linear(%s,%s,%s)" % (str(drqv2_cfg.std0), str(drqv2_cfg.std1), str(drqv2_cfg.std_n_decay))

        self.stddev_clip = drqv2_cfg.stddev_clip
        self.use_data_aug = drqv2_cfg.use_data_aug

        if drqv2_cfg.stage3_update_encoder and drqv2_cfg.encoder_lr_scale > 0:
            self.stage3_update_encoder = True
        else:
            self.stage3_update_encoder = False

        self.act_dim = action_dim

        self.encoder = ActorCriticEncoder(cfg,).to(self.device)
        downstream_input_dim = self.encoder.repr_dim

        self.actor = DrQv2Actor(cfg, downstream_input_dim).to(self.device)
        self.critic: EnsembleCritic = EnsembleCritic(cfg, downstream_input_dim, 
                                                     num_critics=drqv2_cfg.num_critics).to(self.device)
        # self.critic = DoubleQMLP(cfg, downstream_input_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device).requires_grad_(False)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        encoder_lr = lr * drqv2_cfg.encoder_lr_scale
        """ set up encoder optimizer """
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def predict_act(self, obs: torch.Tensor, eval_mode=False, force_action_std=None, step=None, is_resample=False):
        assert obs.ndim == 4  # Ensure observation shape is correct
        obs = self.encoder(obs).flatten(start_dim=1)
        
        step = 1 if eval_mode else step  # When eval, step has no meaning

        if force_action_std == None:
            stddev = utils.schedule(self.stddev_schedule, step)
            if step < self.num_expl_steps and not eval_mode:
                action = (torch.rand(self.act_dim) * 2 - 1).unsqueeze(0).float()
                return action
        else:
            stddev = force_action_std
        # stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor.get_dist(obs, stddev)
        if eval_mode:
            action = dist.mean
            return action
        else:
            action = dist.sample(clip=None)
            return action

    def bc_transfer_ac(self):
        self.stage = 3

    def update(self, rb, step, expertrb):
    # def update(self, batch, step):
        # for stage 2 and 3, we use the same functions but with different hyperparameters
        assert self.stage in (2, 3)
        metrics = dict()

        if self.stage == 3 and step % self.update_every_steps != 0:
            return metrics

        if self.stage == 2:
            bc_weight = self.bc_weight
            utd_ratio = 1
            # batch = expertrb.sample(mini_batch_size=self.mini_batch_size)

            update_encoder = self.stage2_update_encoder
            stddev = self.stage2_std

        elif self.stage == 3:
            bc_weight = 0
            if step <= self.num_expl_steps:
                return metrics

            if self.offline_data_ratio > 0:
                utd_ratio = self.utd_ratio
                collect_batch = next(rb)
                expert_batch = next(expertrb)
                # collect_batch = rb.sample(mini_batch_size=self.mini_batch_size * self.utd_ratio * (1-self.offline_data_ratio))
                # expert_batch = expertrb.sample(mini_batch_size=self.mini_batch_size * self.utd_ratio * self.offline_data_ratio)
                batch = merge_batches(collect_batch, expert_batch)
            else:
                utd_ratio = 1
                batch = rb.sample(mini_batch_size=self.mini_batch_size)

            update_encoder = self.stage3_update_encoder
            stddev = utils.schedule(self.stddev_schedule, step)

        # if self.stage == 2:
        if len(batch) == 6:
            obss, actions, rewards, dones_or_discounts, _, next_obss  = utils.to_torch(batch, device=self.device)
        elif len(batch) == 5:
            obss, actions, rewards, dones_or_discounts, next_obss = utils.to_torch(batch, device=self.device)
        
        if not torch.any(dones_or_discounts == 0.99):
            discounts = (1-dones_or_discounts) * 0.99  # dones
        else: 
            discounts = dones_or_discounts             # discounts

        # augment
        if self.use_data_aug:
            obss = self.aug(obss.float())
            next_obss = self.aug(next_obss.float())
        else:
            obss = obss.float()
            next_obss = next_obss.float()

        for i in range(utd_ratio):

            obs, action, reward, next_obs, discount = self.slice(i, obss, actions, rewards, next_obss, discounts)

            # encode
            if update_encoder:
                obs = self.encoder(obs).flatten(start_dim=1)
            else:
                with torch.no_grad():
                    obs = self.encoder(obs).flatten(start_dim=1)

            with torch.no_grad():
                next_obs = self.encoder(next_obs).flatten(start_dim=1)

            # update critic
            metrics.update(self.update_critic_drqv2(obs, action, reward.float(), discount, next_obs,
                                                stddev, update_encoder,))

        # update actor, following previous works, we do not use actor gradient for encoder update
        metrics.update(self.update_actor_drqv2(obs.detach(), action, stddev, bc_weight))

        metrics['batch_reward'] = reward.mean().item()

        # update critic target networks
        update_exponential_moving_average(self.critic_target, self.critic, self.critic_target_tau)
        return metrics

    def update_critic_drqv2(self, obs, action, reward, discount, next_obs, stddev, update_encoder):
        metrics = dict()

        with torch.no_grad():
            dist = self.actor.get_dist(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_V = self.critic_target.compute_q(next_obs, next_action)
            target_Q = reward + discount *  target_V

        critic_loss = self.critic.compute_loss(obs, action, target_Q)

        # logging
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # if needed, also update encoder with critic loss
        if update_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if update_encoder:
            self.encoder_opt.step()

        return metrics

    def update_actor_drqv2(self, obs, behavor_action, stddev, bc_weight):
        metrics = dict()
        c = self.loss_cfg

        """
        get standard actor loss
        """
        dist = self.actor.get_dist(obs, stddev)
        current_action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(current_action).sum(-1, keepdim=True)
        Q = self.critic.compute_q(obs, current_action)
        actor_loss = -Q.mean()

        """
        combine actor losses and optimize
        """
        if bc_weight > 0:
            # BC loss
            actor_bc_loss = F.mse_loss(current_action, behavor_action)
            lam = bc_weight / Q.detach().abs().mean()
            actor_loss_combined = actor_loss * lam + actor_bc_loss
        else:
            actor_loss_combined = actor_loss

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_combined.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss_combined.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def slice(self, i, obses, actions, rewards, next_obses, dones):
        start_idx = self.mini_batch_size * i
        end_idx = self.mini_batch_size * (i + 1)
        
        obs = obses[start_idx:end_idx]
        action = actions[start_idx:end_idx]
        reward = rewards[start_idx:end_idx]
        next_obs = next_obses[start_idx:end_idx]
        done = dones[start_idx:end_idx]

        return obs, action, reward, next_obs, done