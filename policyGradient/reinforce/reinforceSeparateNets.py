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

class NNet(nn.Module):
    def __init__(self, in_dim, out_dim, hiddenLayers, seed=321):
        super(NNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        architecture = [in_dim] + hiddenLayers + [out_dim]
        block  = [self.__linear_block(in_dim, out_dim) 
                for in_dim, out_dim in zip(architecture, architecture[1:-1])]
        block += [nn.Linear(architecture[-2], architecture[-1])]
        self.layers = nn.Sequential(*block)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x):
        return self.layers(x)

    def __linear_block(self, in_dim, out_dim):
        return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.Dropout(p=0.6),
                nn.ReLU())

    def clearMemory(self):
        raise NotImplementedError

    def update(self, loss):
        raise NotImplementedError

class PolicyEstimator(NNet):
    def __init__(self, in_dim, out_dim, hiddenLayers, seed=1412):
        super(PolicyEstimator, self).__init__(in_dim, out_dim, hiddenLayers, seed)
        self.log_probs = []
        self.rewards = []
        # print("policy's state_dict:")
        # print(self.optimizer.state_dict())

    def forward(self, x):
        out = self.layers(x)
        return F.softmax(out, dim=1)
    
    def update(self, policy_loss):
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.clearMemory()
  
    def clearMemory(self):
        del self.rewards[:]
        del self.log_probs[:]
      
class ValueEstimator(NNet):
    def __init__(self, in_dim, hiddenLayers, seed=123):
        super(ValueEstimator, self).__init__(in_dim, 1, hiddenLayers, seed)
        self.values = []
        # print("value's state_dict:")
        # print(self.optimizer.state_dict())

    def forward(self, x):
        return self.layers(x)

    def clearMemory(self):
        del self.values[:]

    def update(self, value_loss):
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()
        self.clearMemory()

class ReinforceAgent(object):
    def __init__(self, env, episodes=10000,  
            max_steps=1000, seed=1412, render_every=50, log_interval=10):
        self.episodes = episodes
        self.max_steps = max_steps
        self.seed = seed
        self.render_every = render_every
        self.log_interval = log_interval
        self.env = env
        self.env.seed(self.seed)
        self.regularize = True
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = 0.99
        self.regularize = True

        self.valueEstimator = ValueEstimator(self.env.observation_space.shape[0], [128])
        self.policyEstimator = PolicyEstimator(self.env.observation_space.shape[0], self.env.action_space.n, [128])
        #for seeding have to manage the env randomeness and the agent randomness sperately

        #overall history
        self.reward_history = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.alpha = 0.5 
 
    def surrogate_loss(self):
        #another alternative for the loss would be to do collect all in an array
        #and do the loss afterward,
        policy_loss = []
        value_loss = []
        discounted_rewards = self.getDiscountedRewards()
        for log_prob, r, value in zip(self.policyEstimator.log_probs, 
                                      discounted_rewards, 
                                      self.valueEstimator.values):
            #should use item() or deatch() here? whats the diff?
            advantage = r - value.item()
            policy_loss.append(-log_prob * advantage)
            #convert to tensor to allow loss to work
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

        policy_loss, value_loss = torch.stack(policy_loss).sum(), torch.stack(value_loss).sum()
        #loss is a list of tensors with grads, cat makes it a single tensor with
        return policy_loss, value_loss
     
    def select_action(self, state):
        #neural network expects a 2 dimensional tensor that is a float
        #not sure if need unsqeueeze
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.policyEstimator(state), self.valueEstimator(state)
        #get probs from the network
        m = Categorical(probs)
        #sample an action with corresponding probability
        action = m.sample()
        #save the log prob, and state_value
        self.policyEstimator.log_probs.append(m.log_prob(action))
        self.valueEstimator.values.append(state_value)
        return action.item()
    
    def standardize(self, tensor):
        return (tensor - tensor.mean()) / (tensor.std() +  self.eps)
 
    def getDiscountedRewards(self):
        discounted_rewards = []
        reward = 0
        for r in self.policyEstimator.rewards[::-1]:
            reward = r + self.gamma * reward
            discounted_rewards.insert(0, reward)
        #make sure dont need .float()
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        return self.standardize(discounted_rewards) if self.regularize else discounted_rewards
 
    def finish_episode(self):
        #get loss
        policy_loss, value_loss = self.surrogate_loss()
        
        #update history counters, do this before update erases rewards
        self.policy_loss_history.append(policy_loss)
        self.value_loss_history.append(value_loss)
        self.reward_history.append(np.sum((self.policyEstimator.rewards)))
        
        #calc gradients, does backprop on each NN, erases episode arrays
        self.policyEstimator.update(policy_loss)
        self.valueEstimator.update(value_loss)



    def plot(self):
        plt.plot(self.reward_history)
        plt.title("reward history")
        plt.show()
        plt.plot(self.policy_loss_history)
        plt.title("Policy Loss History")
        plt.show()
        plt.plot(self.value_loss_history)
        plt.title("Value Loss History")
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

                self.policyEstimator.rewards.append(reward)

                ep_reward += reward


                if done:
                    break


            self.finish_episode()
            avg_reward = self.calcRunningReward()
            if episode % self.log_interval == 0:
                print(f"Episode: {episode}, Latest Reward: {round(ep_reward, 2)}, Running_Reward: {avg_reward}") 
            
            if avg_reward >= 200:
                print("you have completed the task!")
                self.env.close()
                self.plot()
                break

if __name__ == '__main__':
    # p = PolicyEstimator(8, 4, [128, 64])
    # v = ValueEstimator(8, [128, 64])
    # # print(v)
    # print(p)
    # p = Policy()
    # state = torch.tensor(env.reset()).float()
    # print(state)
    # a = p(state)
    # print(a)
    # print(a[0].dtype)
    env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v0")
    agent = ReinforceAgent(env)
    agent.train()

