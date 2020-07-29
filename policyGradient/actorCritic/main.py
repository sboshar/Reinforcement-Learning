import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from common.models import *
from common.memory import Memory
from IPython.display import clear_output
import matplotlib.pyplot as plt
from common.utils import *
from common.multiprocessing_env import SubprocVecEnv

# def epsilonLogDecay(ep):
#         return max(0, min(1, 1.0 - math.log10((ep  + 1) /  25)))

# def epsilonLinear(self, ep):
#     return self.linearDecay[ep] if (ep < int(self.EPISODES*self.linearFraction)) else self.min_epsilon
    
def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

class ACAgent:
    def __init__(self, env, envs, hidden_size, env_seed=0, torch_seed=0, ANNEAL_LR=True, device="cpu"):
        self.device = device
        self.env = env
        # if env_seed: self.env.seed(env_seed) 
        self.envs = envs    
        self.obs_space  = self.envs.observation_space.shape[0]
        self.action_space = self.envs.action_space.n
        # self.model = ActorCritic(self.obs_space, self.action_space, hidden_size, torch_seed)
        # self.model = PolicyNet(self.obs_space, self.action_space, hidden_size, seed=torch_seed)
        self.model = ActorCritic(4, 2, orthogonal_weights=[True, True], seed=[1,1])
        # self.model = ActorCritic2(4, 2, 128)
        print(self.model)
        self.device = device
        self.num_steps = 10
        self.max_frames = 20_000
        self.lr = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.test_rewards = []
        #expo decay to 10 percent of initial lr
        self.percent = 10
        self.memory = Memory()
        self.gae = True
        self.max_grad_norm = 0

        #manage expo lr and linear decreasing
        if ANNEAL_LR:
            lam = lambda f: 1-f/(self.max_frames/self.num_steps)
            # lam = lambda f: np.exp(-f / ((self.max_frames/self.num_steps) / -np.log(self.percent/100)))
            ps = optim.lr_scheduler.LambdaLR(self.optimizer, 
                                                    lr_lambda=lam)
            # vs = optim.lr_scheduler.LambdaLR(self.val_opt, lr_lambda=lam)
            self.SCHEDULER = ps

            # self.params.VALUE_SCHEDULER = vs

 
    def test_env(self, vis=False):
        state = env.reset()
        if vis: env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.model(state)
            next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            if vis: env.render()
            total_reward += reward
        return total_reward
    
    #right now they all take a random action at the same time it might make sense for them to be
    #independent
    #somehow it learns even when eps is 1.0, which does make much sense to me, 
    # could have to do with value function
    # def get_action_value(self, state, eps=0):
    #     state = torch.FloatTensor(state).to(self.device)
    #     dist, value = self.model(state)
    #     # dist = Categorical(probs)
    #     if random.random() < eps:
    #         action = torch.tensor(np.random.randint(self.action_space, size=len(self.envs)))
    #     else:
    #         action = dist.sample()
    #     return action, dist, value
    def surrogate_loss(self, next_value, entropy):
        if self.gae:
            returns = self.memory.compute_gae(next_value)
        else:
            returns = self.memory.compute_returns(next_value)

        log_probs = torch.cat(self.memory.log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(self.memory.values)

        #not sure if the noramlzation is needed but ive seen it used
        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        return actor_loss + 0.5 * critic_loss - 0.001 * entropy

    # def update(self, next_value):

    def train(self):
        frame_idx = 0
        #why envs.reset() here
        state = envs.reset()
        while frame_idx < self.max_frames:
            # for param_group in self.optimizer.param_groups:
            #     print(param_group['lr'])
            # log_probs = []
            # values    = []
            # rewards   = []
            # masks     = []
            entropy = 0
            # eps = 1.0 - frame_idx  / self.max_frames
            # eps = epsilonLogDecay(frame_idx)
            # print(eps)
            for _ in range(self.num_steps):
                action, value, log_probs, dist_entropy = self.model.act(to_tensor(state, self.device))
                # action, dist, value = self.get_action_value(state, eps=0)
                next_state, reward, done, _ = envs.step(action.cpu().numpy())

                # log_prob = dist.log_prob(action)
                entropy += dist_entropy
                
                self.memory.add(log_probs, 
                                value, 
                                torch.FloatTensor(reward).unsqueeze(1).to(self.device), 
                                torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))
                # log_probs.append(log_prob)
                # values.append(value)
                # rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                # masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))
                
                state = next_state
                frame_idx += 1
                
                if frame_idx % 1000 == 0:
                    print(frame_idx)
                    mean = np.mean([self.test_env() for _ in range(10)])
                    print(mean)
                    self.test_rewards.append(mean)
                    # plot(frame_idx, self.test_rewards)
                    
            
            
            # next_state = torch.FloatTensor(next_state).to(self.device)
            next_value = self.model.get_value(to_tensor(next_state, self.device))
            loss = self.surrogate_loss(next_value, entropy)
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                        self.max_grad_norm)
            self.optimizer.step()
            self.SCHEDULER.step()
            self.memory.clear()

        return self.test_rewards 
        # plot(frame_idx, self.test_rewards)
        

#(self, env, envs, hidden_size, env_seed=0, torch_seed=0, ANNEAL_LR=True, device="cpu"):
if __name__ == "__main__":

    device = get_device()  
    #create a certain number of envs in the vector
    num_envs = 8
    env_name = "CartPole-v0" 

    #figure out env vs envs
    envs = make_envs(env_name, num_envs)
    env = gym.make(env_name)

    #seeds are not working maybe i need to set them for each of the sub envs...
    
    rew = []
    for i in range(1):
        agent = ACAgent(env, envs, 128, device=device)
        a = np.array(agent.train())
        print(len(a))
        rew.append(a)
    for r in rew:
        plt.plot(r)
    plt.show()
    rew = np.mean(np.array(rew), axis=0)
    plt.plot(rew)
    plt.show()




