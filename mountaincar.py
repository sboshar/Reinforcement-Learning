jmport gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()
LEARNING_RATE = 0.1
#measurement of how much we value future reward over curret=nt reward
DISCOUNT = 0.95
EPISODES = 2000

SHOW_EVERY = 500

#higher episilon = greater chance of random action
#decrease epsilon as you go
epsilon = 0.5 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
#print(discrete_os_win_size)

#why those values, upon inspection reward = -1 until reach flag you get 0
# the q table is 3d 20 by 20 by 3
q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
ep_rewards = []
#min is the worst model max is the best model
aggr_ep_reward = {'ep':[], 'avg': [], 'max':[], 'min': []}


def get_discrete_state(state):
# this function makes discrete bins by mapping to continous values into a range of ints
    #print(state)
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))
#print(discrete_state)
for episode in range(EPISODES):
    episode_reward = 0
    if episode % 2000 == 0:
        print(episode)
        render = True
    else:
        render = False
    done = False

    discrete_state = get_discrete_state(env.reset()) 
    while not done:
        #selecting an action
        if np.random.random() <= epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(q_table[discrete_state])
        #executing the action
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        
        #converting new state to a discrete state
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        #hasnt reached the top
        if not done:
            #the max q is the action from the new state that yields the highest q 
            max_future_q = np.max(q_table[new_discrete_state])
            #the crrent q is indexed by (s,a) pair-> (v, acc, a)
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE*(reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        #this is if the position in greater than the goal position
        elif new_state[0] >= env.goal_position:
            #this is the reward for completign thing
            q_table[discrete_state + (action,)] = 0
            print(f"We made it on episode {episode}")

        discrete_state = new_discrete_state
    #per episode update
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    #append the reward to the liat of rewards after each episode
    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        #average reward of the last SHOW_EVERY sample
        average_reward =  np.mean(np.array(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_reward['ep'].append(episode)
        aggr_ep_reward['avg'].append(average_reward)
        #minimum reward in the the last show_every
        aggr_ep_reward['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_reward['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode:{episode} avg: {average_reward} min:{min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")

        

            
env.close()

plt.plot(aggr_ep_reward['ep'], aggr_ep_reward['avg'], label = "avg")
plt.plot(aggr_ep_reward['ep'], aggr_ep_reward['min'], label = "min")
plt.plot(aggr_ep_reward['ep'], aggr_ep_reward['max'], label = "max")
plt.legend(loc=4)
plt.show()
