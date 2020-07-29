import torch
import torch.nn as nn
import gym
from common.multiprocessing_env import SubprocVecEnv

def to_tensor(x, device): 
    return torch.FloatTensor(x).to(device)

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

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/utils.py
def init(module, weight_init, bias_init, gain=1, init_weights=True):
    if init_weights:
        print(module)
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
    return module
#put something like tihs in here??

# def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
#     """Decreases the learning rate linearly"""
#     lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# critic
# init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                         constant_(x, 0))

# self.critic_linear = init_(nn.Linear(hidden_size, 1))