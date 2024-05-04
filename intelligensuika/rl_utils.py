import os
from typing import Any, Callable, Dict, Optional, Type, Union
import gymnasium as gym

from SkipFrameWrapper import SkipMonitor, PhysicsSkipEnv
import datetime
import numpy as np


def make_env(env_id: str, rank: int, seed: int, skip_frame: int, monitor_dir: str = None) -> Callable[[], gym.Env]:
	def _init() -> gym.Env:
		# if the render mode was not specified, we set it to `rgb_array` as default.
		kwargs = {"render_mode": "rgb_array"}
		kwargs.update({"skip_frame": skip_frame})
		env = gym.make(env_id, **kwargs)
		env.action_space.seed(seed + rank)
		# Wrap the env in a Monitor wrapper
		# to have additional training information
		monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
		# Create the monitor folder if needed
		if monitor_path is not None and monitor_dir is not None:
			os.makedirs(monitor_dir, exist_ok=True)
		env = SkipMonitor(env, filename=monitor_path)
		env = PhysicsSkipEnv(env, skip=skip_frame)
		return env

	return _init

# nupdates = 0
#
# def callback(_locals, _globals):
#     global nupdates
#     if nupdates % 10000 == 0:
#         _locals["self"].save(os.path.join("./results", str(nupdates)))
#         print(f"model saved!! : {nupdates}")
#     nupdates += 1
#     return True
