import pickle
from typing import Tuple
import torch
from torch import Tensor
from torch import nn
import re
import numpy as np

def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) if x is not None else x for x in xs)

def compute_returns(rewards, gamma, dones, truncs=None):
    # R = 0
    # returns = []
    # for r in reversed(rewards):
    #     R = r + gamma * R
    #     returns.insert(0, R)
    R = 0
    returns = []
    dones = dones if truncs is None else (dones + truncs).clip(max=1)
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - d)
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

@torch.no_grad()
def compute_lambda_returns(
    rew: Tensor,
    end: Tensor,
    trunc: Tensor,
    val_bootstrap: Tensor,
    gamma: float,
    lambda_: float,
) -> Tensor:
    # import pdb; pdb.set_trace()
    if rew.ndim == 3:
        rew = rew.squeeze(2)
        end = end.squeeze(2)
    assert rew.ndim == 2 and rew.size() == end.size() == trunc.size() == val_bootstrap.size()

    rew = rew.sign()  # clip reward

    end_or_trunc = (end + trunc).clip(max=1)
    not_end = 1 - end
    not_trunc = 1 - trunc

    lambda_returns = rew + not_end * gamma * (not_trunc * (1 - lambda_) + trunc) * val_bootstrap

    if lambda_ == 0:
        return lambda_returns

    last = val_bootstrap[:, -1]
    for t in reversed(range(rew.size(1))):
        lambda_returns[:, t] += end_or_trunc[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns

def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

from torch.distributions.utils import _standard_normal
from torch import distributions as pyd
class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

class OnlineReplayBuffer:
    def __init__(self, capacity, obs_shape, action_shape):
        """
        Initializes the ReplayBuffer.

        Args:
            batch_size (int): The maximum number of transitions to store.
            obs_shape (tuple): Shape of observations (e.g., (3, 128, 128) for images).
            act_dim (int): Dimension of actions.
        """
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Allocate storage for buffer
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
        self.next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
        self.rew = torch.zeros((capacity, 1), dtype=torch.float32)
        self.done = torch.zeros((capacity, 1), dtype=torch.float32)
        self.act = torch.zeros((capacity, *action_shape), dtype=torch.float32)
        self.old_log_prob = torch.zeros((capacity, 1), dtype=torch.float32)
        self.state_value = torch.zeros((capacity, 1), dtype=torch.float32)

    def store(self, obs, next_obs, rew, done, act, old_log_prob=None, state_value=None):
        """
        Stores a transition in the buffer.

        Args:
            obs (torch.Tensor): Observation.
            next_obs (torch.Tensor): Next observation.
            rew (torch.Tensor): Reward.
            done (torch.Tensor): Done flag (1 if episode ended, 0 otherwise).
            trunc (torch.Tensor): Truncated flag (1 if episode truncated, 0 otherwise).
            act (torch.Tensor): Action.
        """
        if len(obs.shape) == 4:
            b, _, _, _ = obs.shape
        else:
            b, _ = obs.shape

        for i in range(b):
            self.obs[self.ptr] = obs[i]
            self.next_obs[self.ptr] = next_obs[i]
            self.rew[self.ptr] = rew[i]
            self.done[self.ptr] = done[i]
            self.act[self.ptr] = act[i]
            self.ptr = (self.ptr + 1) % self.capacity
            if old_log_prob is not None:
                self.old_log_prob[self.ptr] = old_log_prob[i]
            if state_value is not None:
                self.state_value[self.ptr] = state_value[i]

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, mini_batch_size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a mini-batch of transitions.

        Args:
            mini_batch_size (int): Number of transitions to sample.

        Returns:
            obs, act, rew, next_obs, done, old_log_prob, return (tuple): Tuple of tensors containing the sampled transitions.
        """
        if self.size == 0:
            raise ValueError("Buffer is empty, cannot sample.")

        indices = torch.randint(0, self.size, (mini_batch_size,))

        result = (self.obs[indices], self.act[indices], self.rew[indices], self.next_obs[indices], self.done[indices])

        if self.old_log_prob.sum() != 0:
            result += (self.old_log_prob[indices],)
        else:
            result += (None,)

        if self.state_value.sum() != 0:
            result += (self.state_value[indices],)
        else:
            result += (None,)
        # batch = {
        #     'obs': self.obs[indices],
        #     'next_obs': self.next_obs[indices],
        #     'rew': self.rew[indices],
        #     'done': self.done[indices],
        #     'trunc': self.trunc[indices],
        #     'act': self.act[indices]
        # }
        # IF old_log_prob is not zero
        return result

    def clear(self):
        """
        Clears the buffer by resetting the pointer and size.
        """
        self.ptr = 0
        self.size = 0

    def compute_returns(self, gamma=0.99):
        """
        Compute the returns for the rewards in the buffer.

        Args:
            gamma (float): Discount factor for rewards.
        """
        returns = np.zeros_like(self.rew)
        running_return = 0
        for t in reversed(range(self.size)):
            running_return = self.rew[t] + gamma * running_return * (1 - self.done[t])
            returns[t] = running_return
        self.returns = torch.tensor(returns, dtype=torch.float32)
    

class OfflineReplaybuffer:
    def __init__(self, capacity, obs_shape=(3, 84, 84), action_shape=(4,)):
        """
        Initializes the ReplayBuffer.

        Args:
            batch_size (int): The maximum number of transitions to store.
            obs_shape (tuple): Shape of observations (e.g., (3, 128, 128) for images).
            act_dim (int): Dimension of actions.
        """
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Allocate storage for buffer
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.obs_ = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.act = np.zeros((capacity, *action_shape), dtype=np.float32)

        self.returns = None

    def store(self, obs, act, next_obs, rew, done):
        """
        Stores a transition in the buffer.

        Args:
            obs (torch.Tensor): Observation.
            next_obs (torch.Tensor): Next observation.
            rew (torch.Tensor): Reward.
            done (torch.Tensor): Done flag (1 if episode ended, 0 otherwise).
            trunc (torch.Tensor): Truncated flag (1 if episode truncated, 0 otherwise).
            act (torch.Tensor): Action.
        """
        # detect obs
        # image input
        if len(obs.shape) == 3:
            if obs.shape[2] == 3:
                # (128,128,3) -> (3,128,128)
                obs = np.transpose(obs, (2,0,1))
                next_obs = np.transpose(next_obs, (2,0,1))

            if obs.max() > 1:
                obs = ( obs / 255.0 ) * 2 - 1
                next_obs = ( next_obs / 255.0 ) * 2 - 1

        self.obs[self.ptr] = obs
        self.rew[self.ptr] = rew
        self.done[self.ptr] = done
        self.act[self.ptr] = act
        self.obs_[self.ptr] = next_obs
        self.ptr = (self.ptr + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)

    def sample(self, mini_batch_size):
        """
        Samples a mini-batch of transitions and returns as torch tensors.

        Args:
            mini_batch_size (int): Number of transitions to sample.
        Returns:
            Tuple: A tuple of torch tensors (obs, act, rew, done, return, next_obs).
        """
        if self.size == 0:
            raise ValueError("Buffer is empty, cannot sample.")

        indices = np.random.choice(self.size - 1, mini_batch_size, replace=False)

        # Convert numpy arrays to torch tensors
        obs_batch = torch.tensor(self.obs[indices], dtype=torch.float32)
        next_obs_batch = torch.tensor(self.obs_[indices], dtype=torch.float32)
        act_batch = torch.tensor(self.act[indices], dtype=torch.float32)
        rew_batch = torch.tensor(self.rew[indices], dtype=torch.float32)
        done_batch = torch.tensor(self.done[indices], dtype=torch.float32)

        if self.returns is not None:
            returns_batch = torch.tensor(self.returns[indices], dtype=torch.float32)
            return obs_batch, act_batch, rew_batch, done_batch, returns_batch, next_obs_batch
        else:
            return obs_batch, act_batch, rew_batch, done_batch, None, next_obs_batch
        
    def sample_all(self):
        """
        Returns all the transitions stored in the buffer.
        return obs, act, rew, done, next_obs
        """

        return torch.tensor(self.obs[:self.size], dtype=torch.float32), \
            torch.tensor(self.act[:self.size], dtype=torch.float32), \
            torch.tensor(self.rew[:self.size], dtype=torch.float32), \
            torch.tensor(self.done[:self.size], dtype=torch.float32), \
            torch.tensor(self.obs_[:self.size], dtype=torch.float32)

    def clear(self):
        """
        Clears the buffer by resetting the pointer and size.
        """
        self.ptr = 0
        self.size = 0

    def save(self, filepath):
        """
        Saves the replay buffer to a file using pickle, saving only valid data.

        Args:
            filepath (str): Path to save the buffer.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'obs': self.obs[:self.ptr],  # Only save the valid part
                'rew': self.rew[:self.ptr],
                'done': self.done[:self.ptr],
                'act': self.act[:self.ptr],
                "next_obs": self.obs_[:self.ptr]
            }, f)
        print(f"Buffer saved to {filepath}")

    def load(self, filepath):
        """
        Loads the replay buffer from a file using pickle.

        Args:
            filepath (str): Path to load the buffer from.
        """
        # try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            assert len(data['obs']) == len(data['rew']) == len(data['done']) == len(data['act'])
            self.obs[:len(data['obs'])] = data['obs']
            self.rew[:len(data['rew'])] = data['rew']
            self.done[:len(data['done'])] = data['done']
            self.act[:len(data['act'])] = data['act']
            self.obs_[:len(data['next_obs'])] = data['next_obs']
            self.size = len(data['obs'])
            self.ptr = self.size
        print(f"Buffer loaded from {filepath}")
        # except: 
        #     raise FileNotFoundError(f" {filepath} not found. Please check the file path.")

    def compute_returns(self, gamma=0.99):
        """
        Compute the returns for the rewards in the buffer.

        Args:
            gamma (float): Discount factor for rewards.
        """
        returns = np.zeros_like(self.rew)
        running_return = 0
        for t in reversed(range(self.size)):
            running_return = self.rew[t] + gamma * running_return * (1 - self.done[t])
            returns[t] = running_return
        self.returns = returns

    def state_normalizer(self):
        if len(self.obs.shape) == 4:
            raise NotImplementedError("Only support 1D state")
        self.mean = self.obs.mean(axis=0)
        self.std = self.obs.std(axis=0)
        self.obs = (self.obs - self.mean) / (self.std + 1e-8)
        self.obs_ = (self.obs_ - self.mean) / (self.std + 1e-8)

# test
if __name__ == "__main__":
    # test compute_returns
    rewards = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
    gamma = 0.99
    dones = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    truncs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    print(compute_returns(rewards, gamma, dones, truncs))