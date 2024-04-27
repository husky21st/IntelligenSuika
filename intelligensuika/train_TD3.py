import gymnasium as gym
from tqdm import tqdm
import numpy as np
import os

print(f"{gym.__version__}")


from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

import suikaenv


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


class SuikaRL:
    def __init__(self, model_name, num_procs=2):
        self.model_name = model_name
        self.env_name = "suika-v0"
        self.num_procs = num_procs
        self.env = SubprocVecEnv(
            [make_env(self.env_name, i + self.num_procs) for i in range(self.num_procs)],
            start_method="spawn",
        )
        self.model = TD3("MlpPolicy", self.env, verbose=1)

    def train_model(self, timesteps):
        self.model.learn(total_timesteps=timesteps)
        self.model.save(self.model_name)
        # mean_reward, std_reward = evaluate_policy(td3_agent, env, n_eval_episodes=10)
        # print(f"Mean reward = {mean_reward} +/- {std_reward}")

    def predict_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def load_model(self):
        self.model.load(self.model_name)


if __name__ == '__main__':
    model_name = "suikaRL_agent"
    suika_rl = SuikaRL(model_name=model_name, num_procs=os.cpu_count() - 2)
    suika_rl.train_model(timesteps=int(3e3))
    suika_rl.load_model()
