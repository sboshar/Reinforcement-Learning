import gym
import numpy as np
import random
import math
#I wonder if we need both the sin and cos of the angles
env = gym.make('Acrobot-v1')
print(env.observation_space)
print(env.action_space)
print(env.observation_space.high)
print(type(env.observation_space.low))
LOW = env.observation_space.low
HIGH = env.observation_space.high
EPISODES = 1000
buckets = (10,) * 6  
#inital a q table to zero
q = np.zeros(buckets + (env.action_space.n,))
min_epsilon = 0.1
ada_divisor = 25
min_alpha = 0.1
#how much do you care about future rewards
gamma = .9

#kappa > 0 smaller number means more exploration
kappa = 0.5 
#buckets is a tuple  of buckets could make buckets a field, for now i will leave it
def discretize(obs):
  return ((obs - LOW) / (HIGH - LOW) * buckets).astype(int)

#returns a log decreasing epsilon based on the  
def getLogEp(episode):
  return max(min_epsilon, min(1, 1.0 - math.log10((episode  + 1) / ada_divisor)))

#this will be polynomial decreasing but can also try log later
def getAlpha(ep, p=1):
  return max(min_alpha, min(1, kappa / math.pow((kappa + ep), p)))

#pass in getEp which is a function that return epsilon based on the episode
def decEpGreedy(s, episode, getEp):
  return env.action_space.sample() if random.random() < getEp(episode) else np.argmax(q[s])

def updateQ(s, a, nextState, reward, gamma, alpha):
  q[s][a] = (1 - alpha) * q[s][a] + alpha * (reward + gamma * np.max(q[nextState]))

def renderEvery(every, ep):
  if ep % every == 0:
    env.render()
 
def qLearning():
  for ep in range(EPISODES):
    print(ep)
    done = False
    #draw a random state from S amd reset the environment
    currentState = discretize(env.reset())
    #this is an individual trial
    #gam = getGamma()
    alph = getAlpha(ep)
    totalReward = 0
    while not done:
      #select an action
      action = decEpGreedy(currentState, ep, getLogEp)

      #execute the action
      obs, reward, done, info = env.step(action)
      totalReward += reward

      nextState = discretize(obs)
      #update the q table
      updateQ(currentState, action, nextState, reward, gamma, alph)

      renderEvery(20, ep)

      currentState = nextState
    print(totalReward)

if __name__ == "__main__":    
  qLearning()