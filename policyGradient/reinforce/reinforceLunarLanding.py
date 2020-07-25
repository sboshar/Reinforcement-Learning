import argparse
from statistics import mean
from collections import deque
import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, env, hiddenLayers, seed=1412):
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.env = env
        architecture = [self.env.observation_space.shape[0]] + hiddenLayers + [self.env.action_space.n]
        block  = [self.linear_block(in_dim, out_dim) 
                for in_dim, out_dim in zip(architecture, architecture[1:-1])]
        block += [nn.Linear(architecture[-2], architecture[-1])]
        self.layers = nn.Sequential(*block)

    def forward(self, x):
        out = self.layers(x)
        return F.softmax(out, dim=1)

    def linear_block(self, in_dim, out_dim):
        return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=0.6),
                nn.ReLU())

class ReinforceAgent(object):
    def __init__(self, env, hiddenLayers, episodes=10000, gamma=0.99, regularize=True,  
            max_steps=1000, seed=1412, render_every=50, log_interval=10):
        self.gamma = gamma
        self.episodes = episodes
        self.max_steps = max_steps
        self.seed = seed
        self.render_every = render_every
        self.log_interval = log_interval
        self.env = env
        self.env.seed(self.seed)
        self.policy = Policy(self.env, hiddenLayers, self.seed)
        #for seeding have to manage the env randomeness and the agent randomness sperately
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.regularize = regularize

        self.running_reward = deque([], maxlen=100)

        #episode long counters
        self.saved_log_probs = []
        self.rewards = []

        #overall history
        self.reward_history = []
        self.loss_history = []

    def updateQueue(self, ep_reward):
        if len(self.running_reward) >= 100:
            self.running_reward.popleft()
        self.running_reward.append(ep_reward)


    def select_action(self, state):
        #neural network expects a 2 dimensional tensor that is a float
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        #get probs from the network
        m = Categorical(probs)
        #sample an action with corresponding probability
        action = m.sample()
        #save the log prob
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def standardize(self, arr):
        return (arr - arr.mean()) / (arr.std() +  self.eps)
 
    def getDiscountedRewards(self):
        discounted_rewards = []
        reward = 0
        for r in self.rewards[::-1]:
            reward = r + self.gamma * reward
            discounted_rewards.insert(0, reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        return self.standardize(discounted_rewards) if self.regularize else discounted_rewards
 
    def surrogate_loss(self):
        loss = []
        discounted_rewards = self.getDiscountedRewards()
        for log_prob, r in zip(self.saved_log_probs, discounted_rewards):
            loss.append(-log_prob * r)
        #loss is a list of tensors with grads, cat makes it a single tensor with
        #one grad "catgrad"
        return torch.cat(loss).sum()
      
    def finish_episode(self):
        self.optimizer.zero_grad()
        #calls getdiscounterwrds, and returns the surrogate loss 
        loss = self.surrogate_loss()
        loss.backward()

        self.optimizer.step()
        #update counters
        self.loss_history.append(loss)
        self.reward_history.append(np.sum((self.rewards)))
        
        #reset the episode based counters
        self.saved_log_probs = []
        self.rewards = []
  
    def train(self):
        for episode in range(self.episodes): 
            state, ep_reward = self.env.reset(), 0
            for step in range(1, self.max_steps):

                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)  

                if self.render_every and episode % self.render_every == 0:
                    self.env.render()

                self.rewards.append(reward)

                ep_reward += reward


                if done:
                    break


            self.updateQueue(ep_reward)
            self.finish_episode()
            if len(self.reward_history) < 1:
                running_reward = 0
            elif len(self.reward_history) < 100:
                running_reward = mean(self.reward_history)
            else:
                running_reward = mean(self.reward_history[-100:])
            
            if episode % self.log_interval == 0:
                print(f"Episode: {episode}, Loss: {round(float(self.loss_history[-1]), 2)},\
                        Latest Reward: {round(ep_reward, 2)}, Running_Reward: {round(mean(self.running_reward), 2)}") 
            
            if mean(self.running_reward) >= 200:
                print("you have completed the task!")
                self.env.close()
                #maybe plot
                break



            # if episode % self.log_interval == 0:
                # print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}, {:.2f}'.format(
                      # episode, ep_reward, running_reward, mean(self.running_reward)))
            # if running_reward > self.env.spec.reward_threshold:
                # print("Solved! Running reward is now {} and "
                      # "the last episode runs to {} time steps!".format(running_reward, step))
                # self.env.close()
                # break


                

if __name__ == "__main__":
    #in order to solve have to average 200 for 100 consequtive episodes
    env = gym.make("LunarLander-v2")
    # p = Policy(env, [64, 128])
    # print(p)
    agent = ReinforceAgent(env, [128], seed=312, regularize=True, render_every=0)
    agent.train()
    plt.plot(agent.reward_history)
    plt.title("reward history")
    plt.show()
    plt.plot(agent.loss_history)
    plt.title("loss history")
    plt.show()



