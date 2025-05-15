# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from rl_agent.baseagent import ActorCriticConfig, BaseAgent
from rl_agent.utils import update_exponential_moving_average, merge_batches

from rl_agent.net import Actorlog, DoubleQMLP, DrQv2Actor, VRL3Actor, ActorCriticEncoder, EnsembleCritic, CPActor
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

@dataclass
class CP3ERConfig:
    critic_target_tau: float = 0.01
    num_expl_steps: int = 2000          # because there are no offline steps, so need num_expl_steps
    update_every_steps: int = 2
    use_data_aug: bool = True
    encoder_lr_scale: float = 1
    stage2_update_encoder: bool = True # in this version, we will directly online
    stage3_update_encoder: bool = True
    utd_ratio: float = 1
    mini_batch_size: int = 256
    num_critics: int = 2
    offline_data_ratio: float = 0.5
    bc_weight: float = 0.05

# 先验证 Online
class CP3ERAgent(BaseAgent):
    def __init__(self, cfg: ActorCriticConfig, cp3er_config: CP3ERConfig = None, device="cuda"):
        
        super().__init__(cfg)

        # ====== stage 2, 3 ======
        if cp3er_config is None:
            cp3er_config = CP3ERConfig()
        
        # ------------------------- set default values ------------------------
        action_dim = cfg.num_actions
        lr=1e-4
        self.ac_type = "vrl3"
        self.device = device
        self.stage = 2
        # ------------------------- set default values end ------------------------

        self.critic_target_tau = cp3er_config.critic_target_tau
        self.update_every_steps = cp3er_config.update_every_steps
        self.num_expl_steps = cp3er_config.num_expl_steps
        self.utd_ratio = cp3er_config.utd_ratio
        self.offline_data_ratio = cp3er_config.offline_data_ratio
        self.mini_batch_size = cp3er_config.mini_batch_size

        self.stage2_update_encoder = cp3er_config.stage2_update_encoder

        self.bc_weight = cp3er_config.bc_weight
        self.use_data_aug = cp3er_config.use_data_aug

        if cp3er_config.stage3_update_encoder and cp3er_config.encoder_lr_scale > 0:
            self.stage3_update_encoder = True
        else:
            self.stage3_update_encoder = False

        self.act_dim = action_dim

        self.encoder = ActorCriticEncoder(cfg,).to(self.device)
        downstream_input_dim = self.encoder.repr_dim

        self.actor: CPActor = CPActor(cfg, downstream_input_dim, device=self.device).to(self.device)
        self.critic: EnsembleCritic = EnsembleCritic(cfg, downstream_input_dim, 
                                                     num_critics=cp3er_config.num_critics).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device).requires_grad_(False)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        encoder_lr = lr * cp3er_config.encoder_lr_scale
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

    def predict_act(self, obs: torch.Tensor, ):
        assert obs.ndim == 4  # Ensure observation shape is correct
        obs = self.encoder(obs).flatten(start_dim=1)
        action = self.actor(obs)
        return action

    def bc_transfer_ac(self):
        self.stage = 3

    # def update(self, rb, step, expertrb):
    def update(self, batch, step):
        # for stage 2 and 3, we use the same functions but with different hyperparameters
        assert self.stage in (2, 3)
        metrics = dict()

        if self.stage == 3 and (step % self.update_every_steps != 0 or step < self.num_expl_steps):
            return metrics

        if self.stage == 2:
            bc_weight = self.bc_weight
            utd_ratio = 1
            # batch = expertrb.sample(mini_batch_size=self.mini_batch_size)

            update_encoder = self.stage2_update_encoder

        elif self.stage == 3:
            bc_weight = self.bc_weight
            # if self.offline_data_ratio > 0:
            #     utd_ratio = self.utd_ratio
            #     collect_batch = rb.sample(mini_batch_size=self.mini_batch_size * self.utd_ratio * (1-self.offline_data_ratio))
            #     expert_batch = expertrb.sample(mini_batch_size=self.mini_batch_size * self.utd_ratio * self.offline_data_ratio)
            #     batch = merge_batches(collect_batch, expert_batch)
            # else:
            #     utd_ratio = 1
            #     batch = rb.sample(mini_batch_size=self.mini_batch_size)

            update_encoder = self.stage3_update_encoder

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
            metrics.update(self.update_critic_drqv2(obs, action, reward.float(), discount, next_obs, update_encoder,))

        # update actor, following previous works, we do not use actor gradient for encoder update
        metrics.update(self.update_actor_drqv2(obs.detach(), action, bc_weight))

        metrics['batch_reward'] = reward.mean().item()

        # update critic target networks
        update_exponential_moving_average(self.critic_target, self.critic, self.critic_target_tau)
        return metrics

    def update_critic_drqv2(self, obs, action, reward, discount, next_obs, update_encoder):
        metrics = dict()

        with torch.no_grad():
            next_action = self.actor(next_obs)
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

    def update_actor_drqv2(self, obs, behavor_action, bc_weight):
        metrics = dict()
        c = self.loss_cfg

        """
        get standard actor loss
        """
        current_action = self.actor(obs)
        Q = self.critic.compute_q(obs, current_action)
        actor_loss = -Q.mean()

        """
        combine actor losses and optimize
        """
        if bc_weight > 0:
            # BC loss
            # actor_bc_loss = self.actor.loss(behavor_action, obs)
            actor_bc_loss = F.mse_loss(current_action, behavor_action)
            lam = bc_weight
            # lam = bc_weight / Q.detach().abs().mean()
            actor_loss_combined = actor_loss * lam + actor_bc_loss
        else:
            actor_loss_combined = actor_loss

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_combined.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss_combined.item()

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