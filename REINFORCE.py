import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class policy(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim= -1)


class Reinforce:
    def __init__(self,env, learning_rate = 0.01, discount_factor = 0.99, epsilon = 0.1):
        self.env = gym.make(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.s_space = self.env.observation_space.shape[0]
        self.a_space = self.env.action_space.n
        self.hidden = 32
        self.policy = policy(self.s_space,self.hidden, self.a_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def calculate_G(self,rewards):
        G = 0
        G_list = list()
        for r in reversed(rewards):
            G= ( r + self.discount_factor * G)
            G_list.append(G)
        G_list.reverse()
        return G_list

    def select_action(self, state):
        action_prob = self.policy(state)
        dist = Categorical(action_prob) # samples from the softmax dist output
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob # return both so we can use log_prob to optimize for stochastic gradient ascent

    def train(self, episodes):
        episode_rewards = []
        for i in range(episodes):
            rewards = []
            log_probs = []
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype = torch.float32)
            done = False
            trunc = False
            while not (done or trunc):
                action, log_prob = self.select_action(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                state = torch.tensor(next_state, dtype = torch.float32)
                rewards.append(reward)
                log_probs.append(log_prob)

            returns = self.calculate_G(rewards)
            # convert this into tensor format
            returns = torch.tensor(returns, dtype = torch.float32)
            # normalizing for further stability
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            loss = 0
            for log_prob, G in zip(log_probs, returns):
                loss += -log_prob * G # here we update J(theta)
            self.optimizer.zero_grad() # make sure that there are no leftover gradients from previous computation
            loss.backward() # backpropogation through loss
            self.optimizer.step() # gradient ascent for every step
            
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)
            if (i+1) % 10 == 0:
                print(f"Episode {i+1}, Avg Reward of last 10 episodes: {np.mean(episode_rewards[-10:]):.2f}")

        # Plot results
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("REINFORCE Training Performance")
        plt.show()
        return episode_rewards
            

if __name__ == "__main__":
    agent = Reinforce(env = "CartPole-v1")
    rewards = agent.train(episodes=500)
    print("Training complete!")