import pickle
from typing import Tuple
import torch
from torch import Tensor
from torch import nn
import re
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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
    if val_bootstrap.ndim == 1:
        val_bootstrap = val_bootstrap.unsqueeze(1)

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
        self.old_log_prob = torch.zeros((capacity, *action_shape), dtype=torch.float32)
        self.state_value = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dw = torch.zeros((capacity, 1), dtype=torch.float32)

    def store(self, obs, next_obs, rew, done, act, old_log_prob=None, state_value=None, dw=None):
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
            if old_log_prob is not None:
                self.old_log_prob[self.ptr] = old_log_prob[i]
            if state_value is not None:
                self.state_value[self.ptr] = state_value[i]
            if dw is not None:
                self.dw[self.ptr] = dw[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        # import pdb; pdb.set_trace()
        # Update pointer and size
        # self.ptr = (self.ptr + 1) % self.capacity
        # self.size = self.ptr

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

        result += (self.dw[indices],)
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

    def sample_all(self):
        """
        Returns all the transitions stored in the buffer.
        return obs, act, rew, next_obs, done, old_log_prob, state_value, dw
        """
        result = (self.obs[:self.size], self.act[:self.size], self.rew[:self.size], self.next_obs[:self.size], self.done[:self.size])
        if self.old_log_prob.sum() != 0:
            result += (self.old_log_prob[:self.size],)
        else:
            result += (None,)

        if self.state_value.sum() != 0:
            result += (self.state_value[:self.size],)
        else:
            result += (None,)

        result += (self.dw[:self.size],)
        
        return result
        
        # return self.obs[:self.size], self.act[:self.size], self.rew[:self.size], self.done[:self.size], self.next_obs[:self.size]

class OnpolicyOfflineReplaybuffer:
    def __init__(self, capacity, obs_shape=(3, 84, 84), action_shape=(4,), frame_stack=1):
        """
        Initializes the ReplayBuffer.

        Args:
            batch_size (int): The maximum number of transitions to store.
            obs_shape (tuple): Shape of observations (e.g., (3, 128, 128) for images).
            act_dim (int): Dimension of actions.
        """
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.ptr = 0
        self.size = 0

        c, h, w = obs_shape
        if c == 3:
            c = c * frame_stack
        obs_shape = (c, h, w)

        # Allocate storage for buffer
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.obs_ = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.act = np.zeros((capacity, *action_shape), dtype=np.float32)

        self.returns = None

        print("Createing a buffer with capacity: ", capacity)

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
        assert len(obs.shape) == 3 or len(obs.shape) == 4
        assert obs.shape[-1] == obs.shape[-2]
        
        if not (obs.max() <= 1 or obs.min() < 0):
            raise ValueError(f"{obs.max()} and {obs.min()}")

        if len(obs.shape) == 3:
            np.copyto(self.obs[self.ptr], obs)
            np.copyto(self.obs_[self.ptr], next_obs) 
            np.copyto(self.act[self.ptr], act)
            self.rew[self.ptr] = rew
            self.done[self.ptr] = done
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        else:
            b, _, _, _ = obs.shape
            for i in range(b):
                np.copyto(self.obs[self.ptr], obs[i])
                np.copyto(self.obs_[self.ptr], next_obs[i]) 
                np.copyto(self.act[self.ptr], act[i])
                self.rew[self.ptr] = rew[i]
                self.done[self.ptr] = done[i]
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

        indices = np.random.choice(self.size - 1, int(mini_batch_size), replace=False)

        # Convert numpy arrays to torch tensors
        obs_batch = self.obs[indices]
        next_obs_batch = self.obs_[indices]
        act_batch = self.act[indices]
        rew_batch = self.rew[indices]
        done_batch = self.done[indices]

        obs_batch = torch.from_numpy(obs_batch).float()
        next_obs_batch = torch.from_numpy(next_obs_batch).float()
        act_batch = torch.from_numpy(act_batch).float()
        rew_batch = torch.from_numpy(rew_batch).float()
        done_batch = torch.from_numpy(done_batch).float()

        if self.returns is not None:
            returns_batch = self.returns[indices]
            returns_batch = torch.from_numpy(self.returns[indices]).float()
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

        # np.savez_compressed(filepath, 
        #                     obs=self.obs[:self.size],
        #                     next_obs=self.obs_[:self.size],
        #                     act=self.act[:self.size],
        #                     rew=self.rew[:self.size],
        #                     done=self.done[:self.size])
        # print(f"Buffer saved to {filepath}.npz")
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

        # data = np.load(filepath)
        # N = len(data['obs'])
        # self.obs[:N] = data['obs']
        # self.obs_[:N] = data['next_obs']
        # self.rew[:N] = data['rew']
        # self.done[:N] = data['done']
        # self.act[:N] = data['act']
        # self.ptr = N
        # self.size = N
        # print(f"Buffer loaded from {filepath}")

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

class OfflineReplaybuffer:
    def __init__(self, capacity, obs_shape=(3, 84, 84), action_shape=(4,), frame_stack=1):
        """
        Initializes the ReplayBuffer.

        Args:
            batch_size (int): The maximum number of transitions to store.
            obs_shape (tuple): Shape of observations (e.g., (3, 128, 128) for images).
            act_dim (int): Dimension of actions.
        """
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.ptr = 0
        self.size = 0

        c, h, w = obs_shape
        if c == 3:
            c = c * frame_stack
        obs_shape = (c, h, w)

        # Allocate storage for buffer
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.obs_ = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.act = np.zeros((capacity, *action_shape), dtype=np.float32)

        self.returns = None

        print("Createing a buffer with capacity: ", capacity)

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
        assert len(obs.shape) == 3 or len(obs.shape) == 4
        assert obs.shape[-1] == obs.shape[-2]

        # only when offline, the obs is 3D, need to deal
        if obs.max() <= 1. + 1e-5:
            obs = torch.as_tensor(obs, device="cuda").add(1).div(2).mul(255).byte().detach().cpu().numpy()
            next_obs = torch.as_tensor(next_obs, device="cuda").add(1).div(2).mul(255).byte().detach().cpu().numpy()
        
        if obs.max() <= 1 or obs.min() < 0:
            raise ValueError(f"{obs.max()} and {obs.min()}")

        if len(obs.shape) == 3:
            np.copyto(self.obs[self.ptr], obs)
            np.copyto(self.obs_[self.ptr], next_obs) 
            np.copyto(self.act[self.ptr], act)
            self.rew[self.ptr] = rew
            self.done[self.ptr] = done
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        else:
            b, _, _, _ = obs.shape
            for i in range(b):
                np.copyto(self.obs[self.ptr], obs[i])
                np.copyto(self.obs_[self.ptr], next_obs[i]) 
                np.copyto(self.act[self.ptr], act[i])
                self.rew[self.ptr] = rew[i]
                self.done[self.ptr] = done[i]
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

        indices = np.random.choice(self.size - 1, int(mini_batch_size), replace=False)

        # Convert numpy arrays to torch tensors
        obs_batch = self.obs[indices]
        next_obs_batch = self.obs_[indices]
        act_batch = self.act[indices]
        rew_batch = self.rew[indices]
        done_batch = self.done[indices]

        obs_batch = torch.from_numpy(obs_batch).float()
        next_obs_batch = torch.from_numpy(next_obs_batch).float()
        act_batch = torch.from_numpy(act_batch).float()
        rew_batch = torch.from_numpy(rew_batch).float()
        done_batch = torch.from_numpy(done_batch).float()

        if self.returns is not None:
            returns_batch = self.returns[indices]
            returns_batch = torch.from_numpy(self.returns[indices]).float()
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

        # np.savez_compressed(filepath, 
        #                     obs=self.obs[:self.size],
        #                     next_obs=self.obs_[:self.size],
        #                     act=self.act[:self.size],
        #                     rew=self.rew[:self.size],
        #                     done=self.done[:self.size])
        # print(f"Buffer saved to {filepath}.npz")
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

        # data = np.load(filepath)
        # N = len(data['obs'])
        # self.obs[:N] = data['obs']
        # self.obs_[:N] = data['next_obs']
        # self.rew[:N] = data['rew']
        # self.done[:N] = data['done']
        # self.act[:N] = data['act']
        # self.ptr = N
        # self.size = N
        # print(f"Buffer loaded from {filepath}")

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

def merge_batches(batch1, batch2):
    """
    Merges two batches sampled from OfflineReplaybuffer.

    Args:
        batch1 (tuple): The first batch from buffer1.sample().
        batch2 (tuple): The second batch from buffer2.sample().

    Returns:
        tuple: A merged batch (obs, act, rew, done, returns, next_obs)
    """
    merged = []
    for b1, b2 in zip(batch1, batch2):
        if b1 is None or b2 is None:
            merged.append(None)
        else:
            merged.append(torch.cat([b1, b2], dim=0))
    return tuple(merged)

class EfficientReplayBuffer:
    '''Fast + efficient replay buffer implementation in numpy.'''

    def __init__(self, capacity, obs_shape, act_shape, frame_stack=1, nstep=1, discount=0.99,
                 data_specs=None, sarsa=False):
        self.buffer_size = capacity
        self.data_dict = {}
        self.size = 0
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.nstep = nstep
        self.discount = discount
        self.full = False
        self.discount_vec = np.power(discount, np.arange(nstep))  # n_step - first dim should broadcast
        self.next_dis = discount ** nstep
        self.sarsa = sarsa

        self.obs_shape = obs_shape
        self.ims_channels = obs_shape[0] // self.frame_stack
        self.act_shape = act_shape

        self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.float32)
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

    def _initial_setup(self, obs, act):
        self.size = 0

    def add_data_point(self, obs, act, rew, dis, first):
        # first = time_step.first()
        # latest_obs = time_step.observation[-self.ims_channels:]
        latest_obs = obs[-self.ims_channels:]
        if first:
            end_index = self.size + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.size:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.size:end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.size:self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.size:end_index] = latest_obs
                self.valid[self.size:end_invalid] = False
            self.size = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.size], latest_obs)  # Check most recent image
            np.copyto(self.act[self.size], act)
            self.rew[self.size] = rew
            self.dis[self.size] = dis
            self.valid[(self.size + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.size - self.nstep + 1) % self.buffer_size] = True
            self.size += 1
            self.traj_index += 1
            if self.size == self.buffer_size:
                self.size = 0
                self.full = True

    def store(self, obs, act, rew, dis, first):

        if obs.max() > 1:
            raise ValueError(f"obs max {obs.max()} min {obs.min()}")

        if len(obs.shape) == 3:
            self.add_data_point(obs, act, rew, dis, first)
            return

        if first:
            self.add_data_point(obs[0], act, rew, dis, first)

        else:
            for i in range(obs.shape[0]):
                self.add_data_point(obs[i], act[i], rew[i], dis[i], first)

    # def __next__(self, ):
    def sample(self, mini_batch_size):
        indices = np.random.choice(self.valid.nonzero()[0], size=int(mini_batch_size))
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:]  # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        all_rewards = self.rew[gather_ranges]

        # Could implement below operation as a matmul in pytorch for marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        assert obs.shape[2] == obs.shape[3], f"obs shape {obs.shape[3]} != {obs.shape[4]}"

        act = self.act[indices]

        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        obs = torch.from_numpy(obs).float()
        nobs = torch.from_numpy(nobs).float()
        act = torch.from_numpy(act).float()
        rew = torch.from_numpy(rew).float()
        dis = torch.from_numpy(dis).float()

        if self.sarsa:
            nact = self.act[indices + self.nstep]
            return (obs, act, rew, dis, nobs, nact)

        return (obs, act, rew, dis, nobs)

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.size

    def get_train_and_val_indices(self, validation_percentage):
        all_indices = self.valid.nonzero()[0]
        num_indices = all_indices.shape[0]
        num_val = int(num_indices * validation_percentage)
        np.random.shuffle(all_indices)
        val_indices, train_indices = np.split(all_indices,
                                              [num_val])
        return train_indices, val_indices

    def get_obs_act_batch(self, indices):
        n_samples = indices.shape[0]
        obs_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i])
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        act = self.act[indices]
        return obs, act
    
    def save(self, filepath):
        """
        Saves the replay buffer to a file using pickle, saving only valid data.

        Args:
            filepath (str): Path to save the buffer.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'obs': self.obs[:self.size],  # Only save the valid part
                'rew': self.rew[:self.size],
                'dis': self.dis[:self.size],
                'act': self.act[:self.size],
                'valid': self.valid[:self.size],
            }, f)
        print(f"Buffer saved to {filepath}")
    
    def load(self, filepath):
        """
        Loads the replay buffer from a file using pickle.

        Args:
            filepath (str): Path to load the buffer from.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            assert len(data['obs']) == len(data['rew']) == len(data['dis']) == len(data['act'])
            self.obs[:len(data['obs'])] = data['obs']
            self.rew[:len(data['rew'])] = data['rew']
            self.dis[:len(data['dis'])] = data['dis']
            self.act[:len(data['act'])] = data['act']
            self.valid[:len(data['valid'])] = data['valid']
            self.size = len(data['obs'])
        print(f"Buffer loaded from {filepath}")


class EfficientReplayBufferV2:
    '''Fast + efficient replay buffer implementation in numpy.'''

    def __init__(self, capacity, obs_shape, act_shape, frame_stack=1, nstep=1, discount=0.99,
                 data_specs=None, sarsa=False):
        self.buffer_size = capacity
        self.data_dict = {}
        self.size = 0
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.nstep = nstep
        self.discount = discount
        self.full = False
        self.discount_vec = np.power(discount, np.arange(nstep))  # n_step - first dim should broadcast
        self.next_dis = discount ** nstep
        self.sarsa = sarsa

        self.obs_shape = obs_shape
        self.ims_channels = obs_shape[0] // self.frame_stack
        self.act_shape = act_shape

        self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

    def _initial_setup(self, obs, act):
        self.size = 0

    def add_data_point(self, obs, act, rew, dis, first):
        # first = time_step.first()
        # latest_obs = time_step.observation[-self.ims_channels:]
        latest_obs = obs[-self.ims_channels:]
        if first:
            end_index = self.size + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.size:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.size:end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.size:self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.size:end_index] = latest_obs
                self.valid[self.size:end_invalid] = False
            self.size = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.size], latest_obs)  # Check most recent image
            np.copyto(self.act[self.size], act)
            self.rew[self.size] = rew
            self.dis[self.size] = dis
            self.valid[(self.size + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.size - self.nstep + 1) % self.buffer_size] = True
            self.size += 1
            self.traj_index += 1
            if self.size == self.buffer_size:
                self.size = 0
                self.full = True

    def store(self, obs, act, rew, dis, first):

        if obs.max() > 1:
            raise ValueError(f"obs max {obs.max()} min {obs.min()}")

        if len(obs.shape) == 3:
            self.add_data_point(obs, act, rew, dis, first)
            return

        if first:
            self.add_data_point(obs[0], act, rew, dis, first)

        else:
            for i in range(obs.shape[0]):
                self.add_data_point(obs[i], act[i], rew[i], dis[i], first)

    # def __next__(self, ):
    def sample(self, mini_batch_size):
        indices = np.random.choice(self.valid.nonzero()[0], size=int(mini_batch_size))
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:]  # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        all_rewards = self.rew[gather_ranges]

        # Could implement below operation as a matmul in pytorch for marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        assert obs.shape[2] == obs.shape[3], f"obs shape {obs.shape[3]} != {obs.shape[4]}"

        act = self.act[indices]

        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        obs = torch.from_numpy(obs).float()
        nobs = torch.from_numpy(nobs).float()

        act = torch.from_numpy(act).float()
        rew = torch.from_numpy(rew).float()
        dis = torch.from_numpy(dis).float()

        if self.sarsa:
            nact = self.act[indices + self.nstep]
            return (obs, act, rew, dis, nobs, nact)

        return (obs, act, rew, dis, nobs)

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.size

    def get_train_and_val_indices(self, validation_percentage):
        all_indices = self.valid.nonzero()[0]
        num_indices = all_indices.shape[0]
        num_val = int(num_indices * validation_percentage)
        np.random.shuffle(all_indices)
        val_indices, train_indices = np.split(all_indices,
                                              [num_val])
        return train_indices, val_indices

    def get_obs_act_batch(self, indices):
        n_samples = indices.shape[0]
        obs_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i])
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        act = self.act[indices]
        return obs, act
    
    def save(self, filepath):
        """
        Saves the replay buffer to a file using pickle, saving only valid data.

        Args:
            filepath (str): Path to save the buffer.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'obs': self.obs[:self.size],  # Only save the valid part
                'rew': self.rew[:self.size],
                'dis': self.dis[:self.size],
                'act': self.act[:self.size],
                'valid': self.valid[:self.size],
            }, f)
        print(f"Buffer saved to {filepath}")
    
    def load(self, filepath):
        """
        Loads the replay buffer from a file using pickle.

        Args:
            filepath (str): Path to load the buffer from.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            assert len(data['obs']) == len(data['rew']) == len(data['dis']) == len(data['act'])
            self.obs[:len(data['obs'])] = data['obs']
            self.rew[:len(data['rew'])] = data['rew']
            self.dis[:len(data['dis'])] = data['dis']
            self.act[:len(data['act'])] = data['act']
            self.valid[:len(data['valid'])] = data['valid']
            self.size = len(data['obs'])
        print(f"Buffer loaded from {filepath}")

# test
if __name__ == "__main__":
    # test compute_returns
    rewards = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
    gamma = 0.99
    dones = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    truncs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    print(compute_returns(rewards, gamma, dones, truncs))