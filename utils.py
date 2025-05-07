from typing import List, Dict
import wandb
import cv2
import imageio
import numpy as np
import torch

Logs = List[Dict[str, float]]

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, x=None):  # shape:the dimension of input data
        self.n = 0
        if x is not None:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
            self.S = self.std ** 2
        else:
            self.mean = np.zeros(shape)
            self.S = np.zeros(shape)
            self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape, x= None):
        self.running_ms = RunningMeanStd(shape=shape, x=x)

    def __call__(self, x, update=False):
        # Whether to update the mean and std,during the evaluating,update=False
        tensor_flag = False
        if isinstance(x, torch.Tensor):
            x_device = x.device
            x = x.cpu().numpy()
            tensor_flag = True

        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        if tensor_flag:
            x = torch.tensor(x, device=x_device)

        return x

def wandb_log(logs: Logs, epoch: int):
    for d in logs:
        wandb.log({**d}, step=epoch)


class VideoRecorder:
    def __init__(self, root_dir=None, fps=20):

        # create path
        if root_dir is not None:
            self.save_dir = root_dir
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            if obs.max() <= 1 and obs.min() < 0:
                obs = obs.squeeze(0).add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            self.frames.append(obs)

    def save(self, file_name):
        if self.enabled and self.save_dir is not None:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
        elif self.enabled:
            imageio.mimsave(file_name, self.frames, fps=self.fps)

def to_np(xs):
    return tuple(x.squeeze(0).detach().cpu().numpy() for x in xs)


import time

class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time