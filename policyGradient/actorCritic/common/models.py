import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from common.utils import init

class ActorCritic2(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic2, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value

    def get_value(self, x):
        return self(x)[1]
    
    def act(self, x):
        dist, value = self(x)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return action, value, log_probs, dist_entropy

class NNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, seed=0, 
                 activation="Tanh", orthogonal_weights=True):
        super(NNet, self).__init__()
        
        self.orthogonal_weights = orthogonal_weights 
        if seed: self.seed = torch.manual_seed(seed)
        self.activation = activation  
        assert self.activation in ("tanh", "relu"), ("Invalid activation function" +
                                                     "choices are 'Tanh'(default) or 'ReLU")
        self.activation_fn = nn.Tanh()
        if self.activation == 'relu':
            self.activation_fn = nn.ReLU()

        self.architecture = [in_dim] + hidden_layers + [out_dim]
        
        init_ = lambda m: init(m, nn.init.orthogonal_, 
                               lambda x: nn.init.constant_(x, 0), 
                               np.sqrt(2), 
                               self.orthogonal_weights)
        
        block  = [self.__linear_block(in_dim, out_dim, init_) 
                for in_dim, out_dim in zip(self.architecture, self.architecture[1:-1])]
        
        self.body = nn.Sequential(*block)
        
        #initalize the actor head differently, with gain 0.01
        # init_ = lambda m: init(m, nn.init.orthogonal_, 
        #                        lambda x: nn.init.constant_(x, 0), 
        #                        0.01, 
        #                        self.orthogonal_weights)
        
        self.head = nn.Linear(self.architecture[-2], self.architecture[-1])
    
    
    def forward(self, x):
        x = self.body(x)
        return self.head(x)

    #can you just ad nn.init inside of the linear block sequential
    def __linear_block(self, in_dim, out_dim, init_):
        return nn.Sequential(
                init_(nn.Linear(in_dim, out_dim)),
                self.activation_fn)

class ValueNet(NNet):
    def __init__(self, in_dim, hidden_layers, seed=0, activation="tanh", orthogonal_weights=True):
        super(ValueNet, self).__init__(in_dim, 1, hidden_layers, seed, activation, orthogonal_weights)
        init_ = lambda m: init(m, nn.init.orthogonal_, 
                               lambda x: nn.init.constant_(x, 0), 
                               np.sqrt(2), 
                               self.orthogonal_weights)
        #initalize the value head differently from the actor head
        self.head = init_(self.head)
    
    def get_value(self, x):
        return self(x)

class PolicyNet(NNet):
    def __init__(self, in_dim, out_dim, hidden_layers, seed=0, activation="tanh", shared_weights=False, orthogonal_weights=True):
        super(PolicyNet, self).__init__(in_dim, out_dim, hidden_layers, seed, activation, orthogonal_weights)
        self.shared_weights = shared_weights
        #init weights for actor head
        init_ = lambda m: init(m, nn.init.orthogonal_, 
                        lambda x: nn.init.constant_(x, 0), 
                        0.01, 
                        self.orthogonal_weights)
        self.head = init_(self.head)
        
        #if sharing weights create value head with correct init
        if self.shared_weights: 
            init_ = lambda m: init(m, nn.init.orthogonal_, 
                               lambda x: nn.init.constant_(x, 0), 
                               np.sqrt(2), 
                               self.orthogonal_weights)
            self.value_head = init_(nn.Linear(self.architecture[-2], 1))
        
    def forward(self, x):
        x = self.body(x)
        probs = F.softmax(self.head(x), dim=-1)
        if self.shared_weights:
            return Categorical(probs), self.value_head(x)
        return Categorical(probs)
    
    def get_value(self, x):
        assert self.shared_weights, ("This policy Net does not have a value head. " +
                                     "Specifiy shared_weight=True for shared actor/critic network")
        x = self.body(x)
        return self.value_head(x)
    
    def get_probs(self, x):
       x = self.body(x)
       return F.softmax(self.head(x), dim=-1)
    
class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, hidden_layers=[[128], [128]], seed=[0,0], 
                 activation=["tanh", "tanh"], orthogonal_weights=[True, True]):
        super(ActorCritic, self).__init__()
        
        self.actor = PolicyNet(in_dim=obs_space, 
                               out_dim=action_space, 
                               hidden_layers=hidden_layers[0], 
                               seed=seed[0],
                               activation=activation[0],
                               orthogonal_weights=orthogonal_weights[0])
        
        self.critic = ValueNet(in_dim=obs_space, 
                               hidden_layers=hidden_layers[1], 
                               seed=seed[1],
                               activation=activation[1],
                               orthogonal_weights=orthogonal_weights[1])
                               
    def get_value(self, x):
        return self.critic(x)
    
    def get_probs(self, x):
       return F.softmax(self.actor(x), dim=-1)

    def forward(self, x):
        return self.actor(x), self.critic(x)
   
    #takes in a float tensor
    def act(self, x):
        dist, value = self(x)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return action, value, log_probs, dist_entropy


if __name__ == "__main__":
    a = torch.FloatTensor([1,2,3,1])
    p = ActorCritic(4, 2, orthogonal_weights=[True, True], seed=[1,1])
    # p = PolicyNet(4, 2, [128])
    # p = ActorCritic(4, 2, 64) 
    # shared_weights=False)
    print(p)
    print(p(a))
    # print(p.get_probs(a))
    # print(p.get_value(a))
    # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain("relu"))
    
# def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
#     return tf.layers.dense(inputs=inputs,
#                            units=units,
#                            activation=activation_fn,
#                            kernel_initializer=tf.orthogonal_initializer(gain))