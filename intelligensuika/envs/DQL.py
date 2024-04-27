# https://www.dskomei.com/entry/2022/06/09/171712#%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E8%A8%AD%E8%A8%88
from gym_game_field import *
from setting import *
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm

import torch.nn.functional as F

from test import *

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class Actor(nn.Module):
    """
    in: state
    out: actionの平均と対数偏差
    """
    def __init__(self, input_dim, output_dim, hidden_dim, action_scale):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, output_dim)
        self.log_std_linear = nn.Linear(hidden_dim, output_dim)

        self.action_scale = torch.tensor(action_scale)
        self.action_bias = torch.tensor(0.)
        
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.linear1(state))
        x = self.relu(self.linear2(x))
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
    
class Critic(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)  # Ensure the input dimension is correct
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class Agent(object):

    def __init__(self, state_dim, action_dim, action_scale, args, device):

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']

        self.target_update_interval = args['target_update_interval']

        self.device = device

        self.actor_net = Actor(input_dim=state_dim, output_dim=action_dim, hidden_dim=args['hiden_dim'], action_scale=action_scale).to(self.device)
        self.critic_net = Critic(input_dim=state_dim + action_dim, output_dim=1, hidden_dim=args['hiden_dim']).to(self.device)
        self.critic_net_target = Critic(input_dim=state_dim + action_dim, output_dim=1, hidden_dim=args['hiden_dim']).to(self.device)

        hard_update(self.critic_net_target, self.critic_net)
        convert_network_grad_to_false(self.critic_net_target)

        self.actor_optim = optim.Adam(self.actor_net.parameters())
        self.critic_optim = optim.Adam(self.critic_net.parameters())

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if not evaluate:
            action, _ = self.actor_net.sample(state)
        else:
            _, action = self.actor_net.sample(state)
        return action.detach().numpy().reshape(-1)

    def update_parameters(self, memory, batch_size, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_state_action, _ = self.actor_net.sample(next_state_batch)
            next_q_values_target = self.critic_net_target(next_state_batch, next_state_action)
            next_q_values = reward_batch + mask_batch * self.gamma * next_q_values_target

        q_values = self.critic_net(state_batch, action_batch)
        critic_loss = F.mse_loss(q_values, next_q_values)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        action, _ = self.actor_net.sample(state_batch)
        q_values = self.critic_net(state_batch, action)
        actor_loss = - q_values.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_net_target, self.critic_net, self.tau)

        return critic_loss.item(), actor_loss.item()

def soft_update(target_net, source_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target_net, source_net):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data)


def convert_network_grad_to_false(network):
    for param in network.parameters():
        param.requires_grad = False   
    
# 学習によって得られたデータを蓄えておくメモリ
class ReplayMemory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

args = {
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.2,
    'seed': 123456,
    'batch_size': 256,
    'hiden_dim': 256,
    'start_steps': 1000,
    'target_update_interval': 1,
    'memory_size': 100000,
    'epochs': 100,
    'eval_interval': 10
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_dir_path = Path('result')
model_dir_path = Path('model')
if not result_dir_path.exists():
    result_dir_path.mkdir(parents=True)
if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)
    
env = SuikaEnv(render_mode='human')
print(env.observation_space.shape)
print(env.action_space.shape)
print(env.action_space.sample())
agent = Agent(
    state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], action_scale=env.action_space.high[0],
    args=args, device=device
)
memory = ReplayMemory(args['memory_size'])

episode_reward_list = []
eval_reward_list = []

n_steps = 0
n_update = 0
for i_episode in range(1, args['epochs'] + 1):

    episode_reward = 0
    done = False
    state = env.reset()
    while not done:
        
        if args['start_steps'] > n_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        if len(memory) > args['batch_size']:
            agent.update_parameters(memory, args['batch_size'], n_update)
            n_update += 1

        next_state, reward, done, _, _ = env.step(action)
        n_steps += 1
        episode_reward += reward

        memory.push(state=state, action=action, reward=reward, next_state=next_state, mask=float(not done))

        state = next_state

    episode_reward_list.append(episode_reward)

    if i_episode % args['eval_interval'] == 0:
        avg_reward = 0.
        for _  in range(args['eval_interval']):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                with torch.no_grad():
                    action = agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= args['eval_interval']
        eval_reward_list.append(avg_reward)

        print("Episode: {}, Eval Avg. Reward: {:.0f}".format(i_episode, avg_reward))

print('Game Done !! Max Reward: {:.2f}'.format(np.max(eval_reward_list)))

torch.save(agent.actor_net.to('cpu').state_dict(), model_dir_path.joinpath(f'{gym_game_name}_actor.pth'))