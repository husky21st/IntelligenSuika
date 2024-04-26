from gym_game_field import *
from setting import *
import gym
from gym import spaces
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state):
        action_probs = self.actor(state)
        state_values = self.critic(state)
        return action_probs, state_values
def train(model, env, num_episodes, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = model(state)
            action_dist = Normal(action_probs, 0.1)
            action = action_dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            # Calculate loss and update model
            advantage = reward + 0.99 * state_value.item() * (1 - int(done)) - state_value.item()
            actor_loss = -action_dist.log_prob(action) * advantage
            critic_loss = advantage.pow(2)
            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            total_reward += reward
        print(f'Episode {episode}: Total Reward: {total_reward}')
# Initialize the environment and the model
env = SuikaEnv()
state_dim = env.observation_space.shape[0]
action_dim = 1  # Continuous action space
model = ActorCritic(state_dim, action_dim)
# Train the model
train(model, env, num_episodes=1000)