import wandb
from collections import deque
import numpy as np
from torch import is_tensor

__all__ = ['Logger']

class Logger:

    def __init__(self, prefix, maxlen_img=1):
        self.prefix = prefix

        self.other = {}
        self.scal = {}
        self.images = deque([], maxlen=maxlen_img)

    def scalar(self, key, val):
        if key not in self.scal:
            self.scal[key] = []
        if is_tensor(val):
            val = val.item()

        self.scal[key].append(val)

    def scalars(self, **kwargs):
        for key, val in kwargs.items():
            self.scalar(key, val)

    def image(self, image, caption):
        self.images.append((image, caption))

    def other(self, key, val):
        self.other[key] = val

    def reset(self):
        self.scal = {}
        self.other = {}
        self.images.clear()

    def log(self, step, acc='mean'):
        scalar_dict = { f'{self.prefix}/scalars/{key}': getattr(np, acc)(val) for key, val in self.scal.items() }
        image_dict = {  f'{self.prefix}/images/{n}': wandb.Image(img_capt[0], caption=img_capt[1]) for n, img_capt in enumerate(self.images) }
        other_dict = self.other
        try:
            wandb.log({**scalar_dict, **image_dict, **other_dict}, step=step)
        except BrokenPipeError:
            print('Could not log to wandb!')

        self.reset()
