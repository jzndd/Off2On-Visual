# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from models.rl_agent.baseagent import ActorCriticConfig, BaseAgent
from models.rl_agent.utils import update_exponential_moving_average

from models.rl_agent.net import DoubleQMLP, DrQv2Actor
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
from models.rl_agent import  utils

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




class Stage3ShallowEncoder(nn.Module):
    def __init__(self, obs_shape, n_channel):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = n_channel * 35 * 35

        self.n_input_channel = obs_shape[0]
        self.conv1 = nn.Conv2d(obs_shape[0], n_channel, 3, stride=2)
        self.conv2 = nn.Conv2d(n_channel, n_channel, 3, stride=1)
        self.conv3 = nn.Conv2d(n_channel, n_channel, 3, stride=1)
        self.conv4 = nn.Conv2d(n_channel, n_channel, 3, stride=1)
        self.relu = nn.ReLU(inplace=True)

        # TODO here add prediction head so we can do contrastive learning...

        self.apply(utils.weight_init)
        self.normalize_op = transforms.Normalize((0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225))

        self.compress = nn.Sequential(nn.Linear(self.repr_dim, 50), nn.LayerNorm(50), nn.Tanh())
        self.pred_layer = nn.Linear(50, 50, bias=False)

    def transform_obs_tensor_batch(self, obs):
        # transform obs batch before put into the pretrained resnet
        # correct order might be first augment, then resize, then normalize
        # obs = F.interpolate(obs, size=self.pretrained_model_input_size)
        new_obs = obs / 255.0 - 0.5
        # new_obs = self.normalize_op(new_obs)
        return new_obs

    def _forward_impl(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

    def forward(self, obs):
        o = self.transform_obs_tensor_batch(obs)
        h = self._forward_impl(o)
        h = h.view(h.shape[0], -1)
        return h

    def get_anchor_output(self, obs, actions=None):
        # typically go through conv and then compression layer and then a mlp
        # used for UL update
        conv_out = self.forward(obs)
        compressed = self.compress(conv_out)
        pred = self.pred_layer(compressed)
        return pred, conv_out

    def get_positive_output(self, obs):
        # typically go through conv, compression
        # used for UL update
        conv_out = self.forward(obs)
        compressed = self.compress(conv_out)
        return compressed

class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 1
        self.repr_dim = obs_shape[0]

    def forward(self, obs):
        return obs

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.action_shift=0
        self.action_scale=1
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        mu = mu * self.action_scale + self.action_shift
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

    def forward_with_pretanh(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        pretanh = mu
        mu = torch.tanh(mu)
        mu = mu * self.action_scale + self.action_shift
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist, pretanh

class VRL3Agent(BaseAgent):
    def __init__(self, cfg, critic_target_tau=0.01, num_expl_steps=0,
                 update_every_steps=2, stddev_clip=0.3, use_data_aug=True, encoder_lr_scale=1,
                 safe_q_target_factor=0.5, safe_q_threshold=200, pretanh_penalty=5, pretanh_threshold=0.001,
                 stage2_update_encoder=True, cql_weight=1, cql_temp=1, cql_n_random=10, 
                 stage2_std=0.1, stage2_bc_weight=1,
                 stage3_update_encoder=False, std0=0.01, std1=0.01, std_n_decay=500000,
                 stage3_bc_lam0=0, stage3_bc_lam1=0.95):
        
        super().__init__(cfg)

          # ====== stage 2, 3 ======
        
        # ------------------------- set default values ------------------------
        depth = 3
        action_dim = 4
        lr=1e-4
        self.ac_type = "vrl3"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage = 2

        # ------------------------- set default values end ------------------------

        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps

        self.stage2_std = stage2_std
        self.stage2_update_encoder = stage2_update_encoder

        if std1 > std0:
            std1 = std0
        self.stddev_schedule = "linear(%s,%s,%s)" % (str(std0), str(std1), str(std_n_decay))

        self.stddev_clip = stddev_clip
        self.use_data_aug = use_data_aug
        self.safe_q_target_factor = safe_q_target_factor
        self.q_threshold = safe_q_threshold
        self.pretanh_penalty = pretanh_penalty

        self.cql_temp = cql_temp
        self.cql_weight = cql_weight
        self.cql_n_random = cql_n_random

        self.pretanh_threshold = pretanh_threshold

        self.stage2_bc_weight = stage2_bc_weight
        self.stage3_bc_lam0 = stage3_bc_lam0
        self.stage3_bc_lam1 = stage3_bc_lam1

        if stage3_update_encoder and encoder_lr_scale > 0:
            self.stage3_update_encoder = True
        else:
            self.stage3_update_encoder = False

        self.act_dim = action_dim


        downstream_input_dim = self.encoder.repr_dim

        self.actor = DrQv2Actor(cfg, downstream_input_dim).to(self.device)
        self.critic = DoubleQMLP(downstream_input_dim, cfg.feature_dim, action_dim,
                             cfg.hidden_dim, depth).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device).requires_grad_(False)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        encoder_lr = lr * encoder_lr_scale
        """ set up encoder optimizer """
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.train()
        self.critic_target.train()

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
                action = np.random.uniform(0, 1, (self.act_dim,)).astype(np.float32)
                return action
        else:
            stddev = force_action_std

        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
            return action
        else:
            action = dist.sample(clip=None)
            if is_resample:
                return action, None

            return action.detach(), None
    
    def transfer_off2on(self):
        self.stage = 3

    def update(self, rb, step):
        # for stage 2 and 3, we use the same functions but with different hyperparameters
        assert self.stage in (2, 3)
        metrics = dict()
        batch = rb.sample(mini_batch_size=256)

        if self.stage == 2:
            obs, action, reward, dones, _, next_obs  = utils.to_torch(batch, device=self.device)
        else:
            obs, action, reward, next_obs, dones, _, _ = utils.to_torch(batch, device=self.device)

        if self.stage == 2:
            update_encoder = self.stage2_update_encoder
            stddev = self.stage2_std
            conservative_loss_weight = self.cql_weight
            bc_weight = self.stage2_bc_weight

        if self.stage == 3:
            if step % self.update_every_steps != 0:
                return metrics
            update_encoder = self.stage3_update_encoder

            stddev = utils.schedule(self.stddev_schedule, step)
            conservative_loss_weight = 0

            # compute stage 3 BC weight
            bc_data_per_iter = 40000
            i_iter = step // bc_data_per_iter
            bc_weight = self.stage3_bc_lam0 * self.stage3_bc_lam1 ** i_iter

        # augment
        if self.use_data_aug:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        else:
            obs = obs.float()
            next_obs = next_obs.float()

        # encode
        if update_encoder:
            obs = self.encoder(obs).flatten(start_dim=1)
        else:
            with torch.no_grad():
                obs = self.encoder(obs).flatten(start_dim=1)

        with torch.no_grad():
            next_obs = self.encoder(next_obs).flatten(start_dim=1)

        # update critic
        metrics.update(self.update_critic_vrl3(obs, action, reward, dones, next_obs,
                                               stddev, update_encoder, conservative_loss_weight))

        # update actor, following previous works, we do not use actor gradient for encoder update
        metrics.update(self.update_actor_vrl3(obs.detach(), action, stddev, bc_weight,
                                              self.pretanh_penalty, self.pretanh_threshold))

        metrics['batch_reward'] = reward.mean().item()

        # update critic target networks
        update_exponential_moving_average(self.critic, self.critic_target, self.critic_target_tau)
        # utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics

    def update_critic_vrl3(self, obs, action, reward, dones, next_obs, stddev, update_encoder, conservative_loss_weight):
        metrics = dict()
        batch_size = obs.shape[0]

        """
        STANDARD Q LOSS COMPUTATION:
        - get standard Q loss first, this is the same as in any other online RL methods
        - except for the safe Q technique, which controls how large the Q value can be
        """
        with torch.no_grad():
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-dones) * 0.99 * target_V

            if self.safe_q_target_factor < 1:
                target_Q[target_Q > (self.q_threshold + 1)] = self.q_threshold + (target_Q[target_Q > (self.q_threshold+1)] - self.q_threshold) ** self.safe_q_target_factor

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        """
        CONSERVATIVE Q LOSS COMPUTATION:
        - sample random actions, actions from policy and next actions from policy, as done in CQL authors' code
          (though this detail is not really discussed in the CQL paper)
        - only compute this loss when conservative loss weight > 0
        """
        if conservative_loss_weight > 0:
            random_actions = (torch.rand((batch_size * self.cql_n_random, self.act_dim), device=self.device) - 0.5) * 2

            dist = self.actor(obs, stddev)
            current_actions = dist.sample(clip=self.stddev_clip)

            dist = self.actor(next_obs, stddev)
            next_current_actions = dist.sample(clip=self.stddev_clip)

            # now get Q values for all these actions (for both Q networks)
            obs_repeat = obs.unsqueeze(1).repeat(1, self.cql_n_random, 1).view(obs.shape[0] * self.cql_n_random,
                                                                               obs.shape[1])

            Q1_rand, Q2_rand = self.critic(obs_repeat,
                                           random_actions)  # TODO might want to double check the logic here see if the repeat is correct
            Q1_rand = Q1_rand.view(obs.shape[0], self.cql_n_random)
            Q2_rand = Q2_rand.view(obs.shape[0], self.cql_n_random)

            Q1_curr, Q2_curr = self.critic(obs, current_actions)
            Q1_curr_next, Q2_curr_next = self.critic(obs, next_current_actions)

            # now concat all these Q values together
            Q1_cat = torch.cat([Q1_rand, Q1, Q1_curr, Q1_curr_next], 1)
            Q2_cat = torch.cat([Q2_rand, Q2, Q2_curr, Q2_curr_next], 1)

            cql_min_q1_loss = torch.logsumexp(Q1_cat / self.cql_temp,
                                              dim=1, ).mean() * conservative_loss_weight * self.cql_temp
            cql_min_q2_loss = torch.logsumexp(Q2_cat / self.cql_temp,
                                              dim=1, ).mean() * conservative_loss_weight * self.cql_temp

            """Subtract the log likelihood of data"""
            conservative_q_loss = cql_min_q1_loss + cql_min_q2_loss - (Q1.mean() + Q2.mean()) * conservative_loss_weight
            critic_loss_combined = critic_loss + conservative_q_loss
        else:
            critic_loss_combined = critic_loss

        # logging
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # if needed, also update encoder with critic loss
        if update_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss_combined.backward()
        self.critic_opt.step()
        if update_encoder:
            self.encoder_opt.step()

        return metrics

    def update_actor_vrl3(self, obs, action, stddev, bc_weight, pretanh_penalty, pretanh_threshold):
        metrics = dict()

        """
        get standard actor loss
        """
        dist, pretanh = self.actor.forward_with_pretanh(obs, stddev)
        current_action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(current_action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, current_action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()

        """
        add BC loss
        """
        if bc_weight > 0:
            # get mean action with no action noise (though this might not be necessary)
            stddev_bc = 0
            dist_bc = self.actor(obs, stddev_bc)
            current_mean_action = dist_bc.sample(clip=self.stddev_clip)
            actor_loss_bc = F.mse_loss(current_mean_action, action) * bc_weight
        else:
            actor_loss_bc = torch.FloatTensor([0]).to(self.device)

        """
        add pretanh penalty (might not be necessary for Adroit)
        """
        pretanh_loss = 0
        if pretanh_penalty > 0:
            pretanh_loss = pretanh.abs() - pretanh_threshold
            pretanh_loss[pretanh_loss < 0] = 0
            pretanh_loss = (pretanh_loss ** 2).mean() * pretanh_penalty

        """
        combine actor losses and optimize
        """
        actor_loss_combined = actor_loss + actor_loss_bc + pretanh_loss

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_combined.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_loss_bc'] = actor_loss_bc.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        metrics['abs_pretanh'] = pretanh.abs().mean().item()
        metrics['max_abs_pretanh'] = pretanh.abs().max().item()

        return metrics