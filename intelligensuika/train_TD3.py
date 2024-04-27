import gymnasium as gym
from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.append(os.pardir)
from tools.utils import make_dir

print(f"{gym.__version__}")


from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import suikaenv
from utils import *


class SuikaRL:
    def __init__(self):
        self.env_name = "suika-v0"
        # self.num_procs = os.cpu_count() - 4
        self.env = DummyVecEnv([lambda: gym.make(self.env_name)])
        # self.env = SubprocVecEnv(
        #     [make_env(self.env_name, i + self.num_procs) for i in range(self.num_procs)],
        #     start_method="fork",
        # )

        self.model = TD3("MlpPolicy", self.env, verbose=1)

    def train_model(self, timesteps, log_interval):
        self.model.learn(total_timesteps=timesteps, callback=callback, log_interval=int(1e4), progress_bar=True)
        self.model.save(os.path.join("./results", "final"))
        # mean_reward, std_reward = evaluate_policy(td3_agent, env, n_eval_episodes=10)
        # print(f"Mean reward = {mean_reward} +/- {std_reward}")

    def predict_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def load_model(self, id):
        self.model.load(os.path.join("./results", id))


if __name__ == '__main__':
    make_dir("./results")
    suika_rl = SuikaRL()
    suika_rl.train_model(timesteps=int(1e6), log_interval=int(1e4))
    # suika_rl.load_model()
