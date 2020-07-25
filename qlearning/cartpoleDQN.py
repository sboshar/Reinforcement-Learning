import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#create/load the environment unwrapped allows you to access
#the behind the scenes dynamics of the library, possible to remove
#the 200 timestep limit

env = gym.make('CartPole-v0').unwrapped

#set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend())

#I dont think that I need this
if is_python:
    from Ipython import displays

#turns interactive plotting mode on can enter things into terminal and change
#the graph
pl.ion()

#if gpu is bo used otherwise cpu
deivce = tprch.device("cuda" if torch.cuda.is_available() else "cpu")

#REPLAY MEMORY or experience replay
#Transition class uses namedtuple which are cool, represents a dingle transition
# in env, maps (s, a) to (next s, reward)

Transition = namedtuple("transiion", ('state', 'action', 'next_state', 'reward')))

#replay memory class
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = memory
        self.position = 0

    def push(self, *args):
        """saves a Transition"""
        #if memory is shorter than capacity add None to lengthen mem
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        #if mem is still growing, self.pos should be the last pos in the array
        #add new transition obj here
        self.memory[self.position] = Transition(*args)
        #increase self.pos cycliclally
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        #randome list of bs elements
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#DQN algorithm, use a NN as a universal aooriximator for Q
# training update: Q(s) = r + gamma * (s', pi(s'))  Bellman Eq
# temporal dif error delta = Q(s,a) - (r + gamme * max (Q(s', a))
# use huber loss, mse when small , mae when large

#Our model will be a convolutional neural network that takes in the difference between 
#the current and previous screen patches. It has two outputs, representing Q(s,left) 
#and Q(s,right) (where s is the input to the network). In effect, the 
#network is trying to predict the expected return of taking each action given the current input.

class DQN(nn.module):
    def __init__(self, h, e, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernal_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernal_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernal_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        #calc the linear connections
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



def conv_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
            nn.Conv2d(in_f, out_f, *args, **kwargs),
            nn.BatchNorm2d(out_f),
            nn.ReLU()
    )

class DQN2(nn.module):
    def __init__(self, h, e, outputs):
        super(DQN2, self).__init__()
        self.arch = [3, 16, 32, 32]

        #conv layers
        block = [conv_block(in_f, out_f, kernel_size=5, stride=2)
                for in_f, out_f in zip(self.arh, self.arch[1:])]
 

        #calc the linear connections
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
