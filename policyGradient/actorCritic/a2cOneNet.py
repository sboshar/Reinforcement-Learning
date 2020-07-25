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
from memory import Memory
from utils import *

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
    
    def predict_value(self, x):
        x = F.relu(self.affine1(x))
        state_values = self.value_head(x)
        return state_values
    
    def predict_probs(self, x):
        x = F.relu(self.affine1(x))
        return F.softmax(self.action_head(x), dim=-1)



class ActorCritic:
    def __init__(self, env, seed):
        self.env = env
        self.seed = seed
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.policy = Policy()
        self.n_episodes = 500
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.memory = Memory()
        self.reward_history = []
        self.n_step = 10000
        self.gamma = 0.99
        self.regularize = False
        self.eps = np.finfo(np.float32).eps.item()
    
    
    
    def select_action(self, state):
        #neural network expects a 2 dimensional tensor that is a float
        #not sure if need unsqeueeze
        probs = self.policy.predict_probs(to_tensor(state))
        #get probs from the network
        dist = Categorical(probs)
        #sample an action with corresponding probability
        action = dist.sample()
        #save the log prob, and state_value
        return action, dist.log_prob(action)
    
    def standardize(self, tensor):
        return (tensor - tensor.mean()) / (tensor.std() +  self.eps)
 
    def getDiscountedRewards(self):
        discounted_rewards = []
        reward = 0
        for r in self.memory.rewards:
            reward = r + self.gamma * reward
            discounted_rewards.insert(0, reward)
        #make sure dont need .float()
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        return self.standardize(discounted_rewards) if self.regularize else discounted_rewards
    
    def run(self, q_val):
        values = torch.stack(self.memory.values)
        q_vals = np.zeros((len(self.memory), 1))
        
        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        for i, (_, _, reward, done) in enumerate(self.memory.reversed()):
            q_val = reward + self.gamma*q_val*(1.0-done)
            q_vals[len(self.memory)-1 - i] = q_val # store values from the end to the beginning
            
        advantage = torch.Tensor(q_vals) - values
        
        critic_loss = advantage.pow(2).mean()
        self.optimizer.zero_grad()
        actor_loss = (-torch.stack(self.memory.log_probs)*advantage.detach()).mean()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

    def surrogate_loss(self):
        #iterate over memory
        values = torch.stack(self.memory.values)
        discounted_rewards = self.getDiscountedRewards() * ~torch.tensor(self.memory.dones)
        # print(self.memory.dones)
        # print(discounted_rewards)
        advantage = discounted_rewards - values
        actor_loss = (-torch.stack(self.memory.log_probs)*advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        return actor_loss + critic_loss

    def updateNetworks(self, last_val):

        loss = self.surrogate_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        

    def train(self):
        for i in range(self.n_episodes):
            state, ep_reward = self.env.reset(), 0
            done = False
            #add max steps perhaps later
            while not done:
                #select action
                action, log_prob_action = self.select_action(state)
                #take step
                next_state, reward, done, _ = self.env.step(action.item())

                ep_reward += reward  
                #add log_prob, value, reward, and done
                self.memory.add(log_prob_action, self.policy.predict_value(to_tensor(state)), reward, done)
                
                state = next_state
                
                # train if done or num steps > max_steps
                if done or len(self.memory) >= self.n_step:
                    last_val = self.policy.predict_value(to_tensor(next_state)).detach().numpy()
                    # self.updateNetworks(last_val)
                    self.run(last_val)
                    self.memory.clear()
            
                    
            self.reward_history.append(ep_reward)
            if i % 10 == 0:
                print(i, mean(self.reward_history[-10:]))
        
        plt.scatter(np.arange(len(self.reward_history)), self.reward_history, s=2)
        plt.title("Total reward per episode (episodic)")
        plt.ylabel("reward")
        plt.xlabel("episode")
        plt.show()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    seed = 123
    env.seed(seed)
    agent = ActorCritic(env, seed)
    agent.train()