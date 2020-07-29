import torch
import numpy as np

#im a little confused would you ever expect to get a done in the middle of rewards sequence
# does this have to do with batching/randomized minibatching not sure
def calculate_returns(rewards, dones, discount_factor, normalize = True):
    returns = []
    R = 0
    #swap dones so that False (0) now means done/terminal
    for r, d in zip(reversed(rewards), reversed(~dones)):
        R = r + R * discount_factor * d
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns

#I think this is basically the td (lambda) return and then we subtract the values from it to
#get the advantage
#i think that next value refers to the value of the next state from which you predict the values
#of all subsequent states

#values are the predicted values of the states, rewards are the immediate rewards received
#mask are the dones, gamma is discount, lam is the what you you adjust  
def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    return returns

#this expects next value to be at th end of the value list i think
def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])
    return returns

    # adv = np.array(returns) - values[:-1]
    # return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

if __name__ == "__main__":
    # print(compute_returns([1,1,2], [False, True, False]))
    print(calculate_returns([1,1,2], np.array([False, True, False]), 0.9, False))
    print(get_advantages([1,1],[2], 

#advantage function estimators
