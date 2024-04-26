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
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.action_bound  # Scale output to action_bound
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)  # Ensure the input dimension is correct
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        # Ensure action tensor is correctly shaped to concatenate with state
        action = action.unsqueeze(-1) if action.dim() == 1 else action
        print(f"state:{state.shape}, action:{action.shape}")
        combined = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer, replay_buffer, 
                 state_dim, action_dim, action_bound, device, gamma=0.99, tau=0.005):
        self.actor  = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device
        self.gamma = gamma  # Discount factor for future rewards
        self.tau = tau      # Soft update parameter

    def select_action(self, state, noise_scale=0.1):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise_scale > 0:
            action += noise_scale * np.random.randn(self.action_dim)
        return np.clip(action, -self.action_bound, self.action_bound)

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states  = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Update Critic
        next_actions  = self.actor(next_states)
        # print(next_actions.shape)
        # print(next_states.shape)
        next_q_values = self.critic(next_states, next_actions)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        current_q_values  = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(current_q_values, expected_q_values.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update model parameters.
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)

    def soft_update(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
def train(agent,env,n_episodes, batch_size):
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            if len(agent.replay_buffer) >= batch_size:
                agent.update(batch_size)
            # agent.update(batch_size)
        print(f"Episode {episode}: {episode_reward}")

def main():
    env = SuikaEnv(render_mode='rgb-array')
    state_dim    = env.observation_space.shape[1]
    action_dim   = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    print(state_dim, action_dim, action_bound)
    actor = Actor(state_dim, action_dim, action_bound)
    critic = Critic(state_dim, action_dim)
    actor_target = Actor(state_dim, action_dim, action_bound)
    critic_target = Critic(state_dim, action_dim)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    actor_optimizer  = optim.Adam(actor.parameters(), lr=0.0001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
    memory = ReplayBuffer(1000000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(actor, critic, actor_optimizer, critic_optimizer, memory, state_dim, action_dim, action_bound, device)
    train(agent,env,1000, 128)
    env.close()

if __name__ == "__main__":
    main()