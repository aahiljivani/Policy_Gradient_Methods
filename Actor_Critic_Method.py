import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class actor(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim= -1)


class critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1) # for loss computation


class Actor_Critic:
    def __init__(self,env, learning_rate = 0.001, discount_factor = 0.99):
        self.env = gym.make(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.s_space = self.env.observation_space.shape[0]
        self.a_space = self.env.action_space.n
        self.hidden = 64
        self.actor = actor(self.s_space,self.hidden, self.a_space)
        self.critic = critic(self.s_space, self.hidden)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def calculate_delta(self, state, next_state, reward, done, trunc):
        v_state = self.critic(state)
        # Properly handle terminal state!
        if done or trunc:
            v_next_state = torch.tensor(0.0)
        else:
            v_next_state = self.critic(next_state)
        target = reward + self.discount_factor * v_next_state
        delta = target - v_state
        return delta, v_state, target

    def critic_update(self, delta):
        self.critic_optimizer.zero_grad()
        critic_loss = delta.pow(2)
        critic_loss.backward()
        self.critic_optimizer.step()

    def select_action(self, state):
        action_prob = self.actor(state)
        dist = Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def actor_update(self, delta, log_prob, I):
        self.actor_optimizer.zero_grad()
        actor_loss = -delta.detach() * I * log_prob
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self, episodes):
        episode_rewards = []
        for ep in range(episodes):
            state, _ = self.env.reset()
            I = 1
            done, trunc = False, False
            total_reward = 0
            while not (done or trunc):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action, log_prob = self.select_action(state_tensor)
                next_state, reward, done, trunc, _ = self.env.step(action)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                delta, v_state, target = self.calculate_delta(
                    state_tensor, next_state_tensor, reward, done, trunc
                )
                self.critic_update(delta)
                self.actor_update(delta, log_prob, I)
                I *= self.discount_factor
                state = next_state
                total_reward += reward
            episode_rewards.append(total_reward)
            if (ep+1) % 10 == 0:
                print(f"Episode {ep+1}, Avg Reward of last 10 episodes: {np.mean(episode_rewards[-10:]):.2f}")

        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Actor-Critic Training Performance")
        plt.show()
        return episode_rewards

if __name__ == "__main__":
    agent = Actor_Critic(env = "CartPole-v1")
    rewards = agent.train(episodes=1000)
    print("Training complete!")