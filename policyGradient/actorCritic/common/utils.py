import torch
import torch.nn as nn
import gym
from common.multiprocessing_env import SubprocVecEnv

def to_tensor(x): return torch.from_numpy(x).float()

def get_device():
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")

##later make this include gym make as well
def make_envs(env_name, num_envs):
    def make_env():
        def _thunk():
            env = gym.make(env_name)
            return env
        return _thunk
    return SubprocVecEnv([make_env() for i in range(num_envs)])

def normalize(tensor):
    return (tensor - tensor.mean()) / (tensor.std() + 1e-10)






