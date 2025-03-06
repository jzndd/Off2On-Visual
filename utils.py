from typing import List, Dict
import wandb
import cv2
import imageio

Logs = List[Dict[str, float]]


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
            # Convert RGB to BGR for OpenCV compatibility
            if obs.max() <= 1 and obs.min() < 0:
                obs = obs.squeeze(0).add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            self.frames.append(obs)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)