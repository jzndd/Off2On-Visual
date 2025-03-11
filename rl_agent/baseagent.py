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

from torch.distributions import Normal

ActorCriticOutput = namedtuple("ActorCriticOutput", "logits_act")

@dataclass
class ActorCriticLossConfig:
    backup_every: int
    gamma: float
    lambda_: float
    weight_value_loss: float
    weight_entropy_loss: float
    weight_bc_loss: float
    is_continuous_action: bool = True


@dataclass
class ActorCriticConfig:
    feature_dim: int
    hidden_dim: int
    img_size: int
    depth: int = 3
    type: str = None
    num_actions: Optional[int] = None
    num_states:  Optional[int] = None
    is_continuous_action: Optional[bool] = True
    acitive_fn: Optional[str] = "tanh"
    online_lr: Optional[float] = 3e-6
    
class BaseAgent(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()

        # self.encoder = ActorCriticEncoder()
        # self.actor = Actor(cfg, self.encoder.repr_dim)
        # self.critic = Critic(cfg, self.encoder.repr_dim)
        self.feature_dim = cfg.feature_dim
        self.hidden_dim = cfg.hidden_dim
        self.loss_cfg = None
        self.cfg = cfg

    def predict_act(self, obs: Tensor, eval_mode=False) -> None:
        raise NotImplementedError
        # hx, cx = self.lstm(x, hx_cx)
        # return ActorCriticOutput(self.actor_linear(hx), self.critic_linear(hx).squeeze(dim=1), (hx, cx))

    def forward(self, rb, batch = None):
        raise NotImplementedError
    
    def update(self, rb, batch = None):
        raise NotImplementedError
    
    def setup_training(self, loss_cfg: ActorCriticLossConfig) -> None:
        assert self.loss_cfg is None
        self.loss_cfg = loss_cfg