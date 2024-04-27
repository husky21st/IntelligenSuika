import os
import gymnasium as gym
import datetime

import numpy as np
import pytz

from stable_baselines3.common.utils import set_random_seed


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


nupdates = 0


def callback(_locals, _globals):
    global nupdates
    if nupdates % 10000 == 0:
        _locals["self"].save(os.path.join("./results", str(nupdates)))
        print(f"model saved!! : {nupdates}")
    nupdates += 1
    return True
