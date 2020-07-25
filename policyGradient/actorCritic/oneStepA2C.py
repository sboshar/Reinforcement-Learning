import argparse
from statistics import mean
from collections import deque
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# helper function to convert numpy arrays to tensors
def t(x): return torch.from_numpy(x).float()
class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Sequential(nn.Linear(4, 128),
                                  # nn.Dropout(p=0.6), 
                                  # nn.BatchNorm1d(128),
                                  nn.ReLU())

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        #episode buffer for log probs, rewards and values
        self.log_probs = []
        self.rewards = []
        self.values = []

        self.eps = np.finfo(np.float32).eps.item()
        self.regularize = True
        #how much to weight the value losses, 0 means dont take htem into account
        self.alpha = 0.5
        self.smooth_l1 = True
        self.gamma = 0.99

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.seed = torch.manual_seed(123)
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )
    
    def forward(self, X):
        return self.model(X)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.seed = torch.manual_seed(123)
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.model(X)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env.seed(123)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    # policy = Policy()
    actor = Actor(state_dim, n_actions)
    # opt = torch.optim.Adam(policy.parameters(), lr=1e-3) 
    critic = Critic(state_dim)
    adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
    gamma = 0.99
    episode_rewards = []

    for i in range(500):
        if i % 10:
            print(i, episode_rewards[-1])
        done = False
        total_reward = 0
        state = env.reset()


        while not done:
            # probs, current_value = policy(t(state))
            probs = actor(t(state))
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            
            next_state, reward, done, info = env.step(action.detach().data.numpy())
            advantage = reward + (1-done)*gamma* critic(t(next_state))- critic(t(state))
            
            total_reward += reward
            state = next_state

            critic_loss = advantage.pow(2).mean()
            # opt.zero_grad()
            adam_critic.zero_grad()
            critic_loss.backward()
            adam_critic.step()

            actor_loss = -dist.log_prob(action)*advantage.detach()
            adam_actor.zero_grad()
            actor_loss.backward()
            adam_actor.step()
            # loss = actor_loss + critic_loss
            # loss.backward()
            # opt.step()
                
        episode_rewards.append(total_reward)
    plt.scatter(np.arange(len(episode_rewards)), episode_rewards, s=2)
    plt.title("Total reward per episode (online)")
    plt.ylabel("reward")
    plt.xlabel("episode")
    plt.show()




    # states = []
    # for i in range(10):
    #     states.append(env.reset())
    # states = torch.from_numpy(np.array(states)).float()
    # print(states)
    # p = Policy()
    # probs, state_value = p(states)
    # #get probs from the network
    # dist = Categorical(probs)
    # print(dist)
    # #sample an action with corresponding probability
    # action = dist.sample()
    # p