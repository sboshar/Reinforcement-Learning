import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces

class QCartPoleSover:

    def __init__(self, buckets=(1,1,6,12), minV=-0.5, maxV=0.5, 
            minOmega=-math.radians(50), maxOmega=math.radians(50),epsilon=0.25,
            EPISODES=400,gamma=1.0, alphaInit=1, linearDecay=False, min_alpha=0.1,
            min_epsilon=0.1, n_steps=200, ada_divisor=25, threshold=195, linearFraction=0.25):
        self.buckets = buckets
        self.minOmega = minOmega 
        self.maxOmega = maxOmega
        self.maxV = maxV
        self.minV = minV
        self.env = gym.make("CartPole-v0")
        self.epsilon = epsilon
        self.EPISODES = EPISODES
        self.alphaInit = alphaInit
        self.gamma = gamma
        self.min_alpha = min_alpha
        self.min_epsilon = min_alpha
        self.ada_divisor = ada_divisor
        self.rewards = np.array([])
        self.n_steps = n_steps
        self.threshold = threshold
        self.linearFraction = linearFraction

        if linearDecay: self.linearDecay = np.linspace(self.alphaInit, self.min_alpha,int(self.EPISODES*self.linearFraction))
    
        #creates a 6d q table where q[(x, v, th, w)][a] accesses q value
        self.q = np.zeros(self.buckets + (self.env.action_space.n,))



    def discretize(self, obs):
        minObs = np.array([self.env.observation_space.low[0], self.minV, self.env.observation_space.low[2], self.minOmega])
        maxObs = np.array([self.env.observation_space.high[0], self.maxV, self.env.observation_space.high[2],self.maxOmega])
        obsRange = maxObs - minObs
        new_obs = (obs - minObs)/obsRange * self.buckets
        #try and clean up this line later
        new_obs = np.array([min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))])
        return tuple(np.floor(new_obs).astype(int))

    #can also add an min alpha number
    def alphaExpoDecay(self, ep):
        return self.alphaInit / (self.alphaInit + ep)  
    
    def alphaLogDecay(self, ep):
        return max(self.min_alpha, min(1, 1.0 - math.log10((ep  + 1) / self.ada_divisor)))

    #linearly decreasing learning rate 
    def alphaLinear(self, ep):
        return self.linearDecay[ep] if (ep < int(self.EPISODES*self.linearFraction)) else self.min_alpha

    #adaptive/cyclice learning rate-check thatarticle from fastai https://www.jeremyjordan.me/nn-learning-rate/
    def alphaAdaptive(self, ep, cycle):
        pass
    #add simlar things for epsilon
    def epsilonLogDecay(self, ep):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((ep  + 1) / self.ada_divisor)))

    def epsilonLinear(self, ep):
        return self.linearDecay[ep] if (ep < int(self.EPISODES*self.linearFraction)) else self.min_epsilon

    def epsilonGreedy(self, s, epsilon):
        #take random action with prob epsilon
        if random.random() <= epsilon:
            return self.env.action_space.sample()
        #take the best action, which is the action that maximizes q for a given state
        return np.argmax(self.q[s])

    def greedy(self, s):
        return np.argmax(self.q[s])

    def updateQ(self, currentState, action, newState, reward, alpha, gamma):
        #print("CS", currentState)
        #print(newState)
        #print(action)
        #self.q[currentState][action] = (1 - alpha) * self.q[currentState][action] + alpha * (reward + gamma * np.max(self.q[newState]))
        self.q[currentState][action] += alpha * (reward + gamma * np.max(self.q[newState]) - self.q[currentState][action])
            
    def run(self):
        completed = False
        ep = 0
        #while not completed:
        for ep in range(self.EPISODES):
            if ep  % 20 == 0:
                print(ep)
                render = True
            else:
                render = False
            done = False
            #this resets the environment and returns the starting state
            currentState = self.discretize(self.env.reset())
            #start without decay
            gamma = self.gamma
            epsilon = self.epsilonLogDecay(ep)
            alpha = self.alphaLogDecay(ep)
            print(epsilon, alpha)
            #print(epsilon, alpha)
            episode_rewards = 0
            #i am a little confused whether this should dun until done, or until a number of steps
            while not done:
            #for t in range(self.n_steps):
               #select an action
                action = self.epsilonGreedy(currentState, epsilon)
                #execute that action
                obs, reward, done, info = self.env.step(action)
                newState = self.discretize(obs)
                #if not done we want to update q
                #if not done:
                    #we want to pass in the gamma and the alpha that have been change by our functions
                self.updateQ(currentState, action, newState, reward, alpha, gamma)
                if render:
                    self.env.render()
                currentState = newState   

                episode_rewards += reward


                if done:
                    print('Episode:{}/{} finished with a total reward of: {}'.format(ep, self.EPISODES, episode_rewards))
                    break
            self.rewards = np.append(self.rewards, episode_rewards)
            if np.mean(self.rewards[-100:]) > self.threshold and ep >= 100:
                print('Ran {} episodes. Solved after {} trials ✔'.format(ep, ep - 100))
                self.env.close()
                return ep - 100

            ep += 1
        return ep


if __name__ == "__main__":
    solver = QCartPoleSover()
    print(solver.run())
    solver = QCartPoleSover(linearFraction = .4, EPISODES=10000, linearDecay=True, alphaInit=1, min_alpha=.1, min_epsilon=.1) 
    print(solver.run())
    #25 and .1 for log decay
    alphaInit = [1, .9, .8 ]
    min_alpha = [0.075, 0.1, 0.125]
    scores = []
    index = []
    for a in alphaInit:
        for e in min_alpha:
            tempScore = []
            for i in range(5):
                solver = QCartPoleSover(EPISODES=10000, linearDecay=True, alphaInit=a, min_alpha=e) 
                tempScore.append(solver.run())
                #print(tempScore)
            scores.append(np.mean(np.array(tempScore)))
            index.append((a, e ))
            print(tempScore)
            print((a,e))
            print(np.mean(np.array(tempScore)))
    i =  np.argmax(np.array(scores))
    print(index[i])
    print(np.amax(np.array(scores)))
