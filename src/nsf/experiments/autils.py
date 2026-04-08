import math
import os
import time

import torch
from torch import nn
from torch.utils import checkpoint

import autoroot # for imports from src
import src.nsf.utils as utils
from src.nsf.utils import NoDataRootError
from src.nsf.nde import transforms

def imshow(image, ax, vmin=None, vmax=None):
    image = utils.tensor2numpy(image.permute(1, 2, 0))

    # use colormap for single channel images
    if image.shape[-1] == 1:
        im = ax.imshow(image[..., 0], cmap="Blues", vmin=vmin, vmax=vmax)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        im = ax.imshow(image)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_xticklabels("")
    ax.set_yticklabels("")


def get_dataset_root():
    env_var = "DATASET_ROOT"
    try:
        return os.environ[env_var]
    except KeyError:
        raise NoDataRootError(f"Environment variable {env_var} doesn't exist.")


def eval_log_density(log_prob_fn, data_loader, num_batches=None):
    with torch.no_grad():
        total_ld = 0
        batch_counter = 0
        for batch in data_loader:
            if isinstance(batch, list):  # If labelled dataset, ignore labels
                batch = batch[0]
            log_prob = log_prob_fn(batch)
            total_ld += torch.mean(log_prob)
            batch_counter += 1
            if (num_batches is not None) and batch_counter == num_batches:
                break
        return total_ld / batch_counter


def eval_log_density_2(log_prob_fn, data_loader, c, h, w, num_batches=None):
    with torch.no_grad():
        total_ld = []
        batch_counter = 0
        for batch in data_loader:
            if isinstance(batch, list):  # If labelled dataset, ignore labels
                batch = batch[0]
            log_prob = log_prob_fn(batch)
            total_ld.append(log_prob)
            batch_counter += 1
            if (num_batches is not None) and batch_counter == num_batches:
                break
        total_ld = torch.cat(total_ld)
        total_ld = nats_to_bits_per_dim(total_ld, c, h, w)
        return total_ld.mean(), 2 * total_ld.std() / total_ld.shape[0]


class CheckpointWrapper(transforms.Transform):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, inputs):
        return checkpoint.checkpoint(self.transform, inputs)

    def inverse(self, inputs):
        return self.transform.inverse(inputs)


class Conv2dSameSize(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        same_padding = (
            kernel_size // 2
        )  # Padding that would keep the spatial dims the same
        super().__init__(in_channels, out_channels, kernel_size, padding=same_padding)


def descendants_of_type(transform, type):
    if isinstance(transform, type):
        return [transform]
    elif isinstance(transform, transforms.CompositeTransform) or isinstance(
        transform, transforms.MultiscaleCompositeTransform
    ):
        l = []
        for t in transform._transforms:
            l.extend(descendants_of_type(t, type))
        return l
    else:
        return []


class Timer:
    def __init__(self, print=False):
        self.print = print

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.print:
            print(f"Operation took {self.interval:.03f} sec.")


# From https://github.com/tqdm/tqdm/blob/master/tqdm/_tqdm.py
def format_interval(t):
    """
    Formats a number of seconds as a clock time, [H:]MM:SS
    Parameters
    ----------
    t  : int
        Number of seconds.
    Returns
    -------
    out  : str
        [H:]MM:SS
    """
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def progress_string(elapsed_time, step, num_steps):
    rate = step / elapsed_time
    if rate > 0:
        remaining_time = format_interval((num_steps - step) / rate)
    else:
        remaining_time = "..."
    elapsed_time = format_interval(elapsed_time)
    return f"{elapsed_time}<{remaining_time}, {rate:.2f}it/s"


class LogProbWrapper(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, inputs, context=None):
        return self.flow.log_prob(inputs, context)


def nats_to_bits_per_dim(nats, c, h, w):
    return nats / (math.log(2) * c * h * w)


# https://stackoverflow.com/questions/431684/how-do-i-change-directory-cd-in-python/13197763#13197763
class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
