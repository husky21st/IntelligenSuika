import gymnasium as gym
from tqdm import tqdm
import numpy as np
import os
import sys

from stable_baselines3 import TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

import suikaenv
from rl_utils import make_env
from tools.utils import make_dir
sys.path.append(os.pardir)
print(f"{gym.__version__}")


class SuikaRL:
	def __init__(self, model_name: str, result_dir: str = "./results", num_envs: int = 6):
		self.env_id = "suika-v0"
		self.model_name = model_name
		self.result_dir = result_dir
		self.seed = 42
		self.skip_frame = 400
		self.num_envs = num_envs

		self.saved_dir = os.path.join(result_dir, self.model_name)
		self.new_logger = configure(self.saved_dir, ["stdout", "csv", "log"])
		make_dir(self.saved_dir)
		self.env = SubprocVecEnv([make_env(
			self.env_id, i, seed=self.seed, skip_frame=self.skip_frame, monitor_dir=self.saved_dir
		) for i in range(self.num_envs)], start_method="spawn")
		self.env.seed(self.seed)
		# self.env = make_vec_env(self.env_id, env_kwargs={"skip_frame": self.skip_frame},
		#                         n_envs=self.num_envs, seed=42, monitor_dir=self.saved_dir,
		#                         wrapper_class=PhysicsSkipEnv, wrapper_kwargs={"skip": self.skip_frame},
		#                         # vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "spawn"},
		#                         )

		self.model = TD3("MlpPolicy", self.env, verbose=2)
		self.model.set_logger(self.new_logger)

	def train_model(self, learn_timesteps, log_interval):
		self.model.learn(total_timesteps=learn_timesteps, log_interval=log_interval, progress_bar=True)
		self.model.save(os.path.join(self.saved_dir, self.model_name))

	def predict_action(self, obs):
		action, _ = self.model.predict(obs, deterministic=True)
		return action

	def load_model(self):
		self.model.load(os.path.join(self.result_dir, self.model_name))


if __name__ == '__main__':
	result_dir = "./results"
	make_dir(result_dir)
	suika_rl = SuikaRL(model_name="test3", num_envs=1)
	suika_rl.train_model(learn_timesteps=1e5, log_interval=1)
