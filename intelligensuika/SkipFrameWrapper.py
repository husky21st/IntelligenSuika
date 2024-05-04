import gymnasium as gym
import numpy as np
import os
import sys
import time
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

sys.path.append(os.pardir)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymStepReturn
from gymnasium.core import ActType, ObsType


class SkipMonitor(Monitor):
	def __init__(
			self,
			env: gym.Env,
			filename: Optional[str] = None,
			allow_early_resets: bool = True,
			reset_keywords: Tuple[str, ...] = (),
			info_keywords: Tuple[str, ...] = (),
			override_existing: bool = True,
	):
		super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords, override_existing)

	def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
		"""
		Step the environment with the given action

		:param action: the action
		:return: observation, reward, terminated, truncated, information
		"""
		if self.needs_reset:
			raise RuntimeError("Tried to step environment that needs reset")
		observation, reward, terminated, truncated, info = self.env.step(action)
		if info["get_reward"]:
			self.rewards.append(float(reward))
			self.total_steps += 1
		if terminated or truncated:
			self.needs_reset = True
			ep_rew = sum(self.rewards)
			ep_len = len(self.rewards)
			ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
			for key in self.info_keywords:
				ep_info[key] = info[key]
			self.episode_returns.append(ep_rew)
			self.episode_lengths.append(ep_len)
			self.episode_times.append(time.time() - self.t_start)
			ep_info.update(self.current_reset_info)
			if self.results_writer:
				self.results_writer.write_row(ep_info)
			info["episode"] = ep_info
		return observation, reward, terminated, truncated, info


class PhysicsSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
	def __init__(self, env: gym.Env, skip: int = 4) -> None:
		super().__init__(env)
		assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
		assert env.observation_space.shape is not None, "No shape defined for the observation space"
		self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
		self._skip = skip

	def step(self, action: int) -> GymStepReturn:
		last_reward = 0.0
		terminated = truncated = False
		info = {}
		for i in range(self._skip):
			obs, reward, terminated, truncated, info = self.env.step(action)
			done = terminated or truncated
			if i == self._skip - 1:
				self._obs_buffer = obs
			last_reward = float(reward)
			if done:
				break

		return self._obs_buffer, last_reward, terminated, truncated, info


