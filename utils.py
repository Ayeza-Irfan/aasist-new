"""
Utilization functions
"""

import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value {}'.format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras style"""
    return 1.0 / (1.0 + decay * step)


def _get_optimizer(model_parameters, optim_config):
    """Defines an optimizer"""
    if optim_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model_parameters,
                               lr=optim_config['base_lr'],
                               weight_decay=optim_config['weight_decay'])
    elif optim_config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model_parameters,
                                lr=optim_config['base_lr'],
                                weight_decay=optim_config['weight_decay'])
    elif optim_config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model_parameters,
                              lr=optim_config['base_lr'],
                              momentum=optim_config['momentum'],
                              weight_decay=optim_config['weight_decay'])
    else:
        raise ValueError('optimizer error, got:{}'.format(
            optim_config['optimizer']))

    return optimizer


def _get_scheduler(optimizer, optim_config):
    """Defines a scheduler"""
    if optim_config['scheduler'] == 'cosine':
        # epochs * steps_per_epoch -> total_steps
        total_steps = optim_config['epochs'] * optim_config['steps_per_epoch']
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                optim_config['lr_min'] / optim_config['base_lr']))

    elif optim_config['scheduler'] == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: keras_decay(step))
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    """Defines an optimizer and a scheduler"""
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, config = None):
    """ 
    set initial seed for reproduction
    """
    if config is None:
        raise ValueError("config should not be None")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)