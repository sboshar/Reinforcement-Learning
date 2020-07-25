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
    
    def __standardize(self, tensor):
        return (tensor - tensor.mean()) / (tensor.std() +  self.eps)
 
    def __getDiscountedRewards(self):
        discounted_rewards = []
        reward = 0
        for r in self.rewards[::-1]:
            reward = r + self.gamma * reward
            discounted_rewards.insert(0, reward)
        #make sure dont need .float()
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        return self.__standardize(discounted_rewards) if self.regularize else discounted_rewards
 
    def surrogate_loss(self):
        #another alternative for the loss would be to do collect all in an array
        #and do the loss afterward,
        policy_loss = []
        value_loss = []
        print(self.e)
        discounted_rewards = self.__getDiscountedRewards()
        for log_prob, r, value in zip(self.log_probs, discounted_rewards, self.values):
            #should use item() or deatch() here? whats the diff?
            advantage = r - value.item()
            policy_loss.append(-log_prob * advantage)
            if self.smooth_l1:
                # print(torch.tensor([r]))
                # print(value)
                #convert to tensor to allow loss to work
                value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))
            else:
                value_loss.append(F.mse_loss(r, value))

        loss = (1 - self.alpha) * torch.stack(policy_loss).sum() + self.alpha * torch.stack(value_loss).sum()
        #loss is a list of tensors with grads, cat makes it a single tensor with
        return loss
     

    def clearMemory(self):
        del self.log_probs[:]
        del self.values[:]
        del self.rewards[:]


class ReinforceAgent(object):
    def __init__(self, env, episodes=10000,  
            max_steps=1000, seed=1412, render_every=0, log_interval=10):
        self.episodes = episodes
        self.max_steps = max_steps
        self.seed = seed
        self.render_every = render_every
        self.log_interval = log_interval
        self.env = env
        self.env.seed(self.seed)
        self.policy = Policy()
        #for seeding have to manage the env randomeness and the agent randomness sperately
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

        #overall history
        self.reward_history = []
        self.loss_history = []

    def select_action(self, state):
        #neural network expects a 2 dimensional tensor that is a float
        #not sure if need unsqeueeze
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.policy(state)
        #get probs from the network
        m = Categorical(probs)
        #sample an asction with corresponding probability
        action = m.sample()
        #save the log prob, and state_value
        self.policy.log_probs.append(m.log_prob(action))
        self.policy.values.append(state_value)
        return action.item()
 
    def updateNetwork(self):
        self.optimizer.zero_grad()
        #calls getdiscounterwrds, and returns the surrogate loss 
        loss = self.policy.surrogate_loss()
        loss.backward()

        self.optimizer.step()

        #update history counters
        self.loss_history.append(loss)
        self.reward_history.append(np.sum((self.policy.rewards)))
        
        #reset the episode based counters
        self.policy.clearMemory()

    def plot(self):
        # plt.plt(self.reward_history)
        plt.scatter(np.arange(len(self.reward_history)), self.reward_history, s=2)
        plt.title("reward history")
        plt.show()
        plt.plot(self.loss_history)
        plt.title("loss history")
        plt.show()


    def calcRunningReward(self):
        if len(self.reward_history) < 100:
             return round(mean(self.reward_history), 2)
        return round(mean(self.reward_history[-100:]), 2)
    
        
    def train(self):
        for episode in range(self.episodes): 
            state, ep_reward = self.env.reset(), 0
            for step in range(1, self.max_steps):
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)  
                if self.render_every and episode % self.render_every == 0:
                    self.env.render()

                self.policy.rewards.append(reward)

                ep_reward += reward
                if done:
                    break
            
            self.updateNetwork()
            avg_reward = self.calcRunningReward()
            if episode % self.log_interval == 0:
                print(f"Episode: {episode}, Loss: {round(float(self.loss_history[-1]), 2)},\
                        Latest Reward: {round(ep_reward, 2)}, Running_Reward: {avg_reward}") 
            
            # if avg_reward >= 200 or episode >= 800:
            if episode >= 500:
                print("you have completed the task!")
                self.env.close()
                self.plot()
                break



if __name__ == '__main__':
    # p = Policy()
    # state = torch.tensor(env.reset()).float()
    # print(state)
    # a = p(state)
    # print(a)
    # print(a[0].dtype)
    env = gym.make("CartPole-v1")
    agent = ReinforceAgent(env)
    agent.train()

