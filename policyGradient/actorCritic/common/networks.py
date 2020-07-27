import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class NNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, seed=0, 
                 activation="Tanh", orthogonal_weights=True):
        super(NNet, self).__init__()
        if seed: self.seed = torch.manual_seed(seed)
        self.activation = activation  
        assert self.activation in ("tanh", "relu"), ("Invalid activation function" +
                                                     "choices are 'Tanh'(default) or 'ReLU")
        self.activation_fn = nn.Tanh()
        if self.activation == 'relu':
            self.activation_fn = nn.ReLU()
        
        self.architecture = [in_dim] + hidden_layers + [out_dim]
        block  = [self.__linear_block(in_dim, out_dim) 
                for in_dim, out_dim in zip(self.architecture, self.architecture[1:-1])]
        self.body = nn.Sequential(*block)
        self.head = nn.Linear(self.architecture[-2], self.architecture[-1])

        if orthogonal_weights:
            self.body.apply(self.init_weights)
            torch.nn.init.orthogonal_(self.head.weight, 
                                      torch.nn.init.calculate_gain(self.activation))
            self.head.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = self.body(x)
        return self.head(x)

    def __linear_block(self, in_dim, out_dim):
        return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.Dropout(p=0.6),
                self.activation_fn)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.orthogonal_(m.weight, 
                                      torch.nn.init.calculate_gain(self.activation))
            m.bias.data.fill_(0.0)


class ValueNet(NNet):
    def __init__(self, in_dim, hidden_layers, seed=0, activation="tanh", orthogonal_weights=True):
        super(ValueNet, self).__init__(in_dim, 1, hidden_layers, seed, activation, orthogonal_weights)
    
    def get_value(self, x):
        return self(x)

#make a base network class here
class PolicyNet(NNet):
    def __init__(self, in_dim, out_dim, hidden_layers, seed=0, activation="tanh", shared_weights=True, orthogonal_weights=True):
        super(PolicyNet, self).__init__(in_dim, out_dim, hidden_layers, seed, activation, orthogonal_weights)
        self.shared_weights = shared_weights
        if self.shared_weights: self.value_head = nn.Linear(self.architecture[-2], 1)
        if orthogonal_weights:
            torch.nn.init.orthogonal_(self.value_head.weight, 
                                        torch.nn.init.calculate_gain(self.activation))
            self.value_head.bias.data.fill_(1.0)

    def forward(self, x):
        x = self.body(x)
        probs = F.softmax(self.head(x), dim=-1)
        if self.shared_weights:
            return probs, self.value_head(x)
        return probs 
    
    def get_value(self, x):
        assert self.shared_weights, ("This policy Net does not have a value head. " +
                                     "Specifiy shared_weight=True for shared actor/critic network")
        x = self.body(x)
        return self.value_head(x)
    
    def get_probs(self, x):
       x = self.body(x)
       return F.softmax(self.head(x), dim=-1)
    
# notes, add:
# orthogonal initialization, layer scaling, add xavier weights as well
#value f

if __name__ == "__main__":
    a = torch.FloatTensor([1,2,3,1])
    p = PolicyNet(4, 2, [2, 4], shared_weights=True, seed=10, activation="tanh")
    # p = ActorCritic(4, 2, 64) 
    # shared_weights=False)
    print(p(a))
    # print(p.get_probs(a))
    # print(p.get_value(a))
    # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain("relu"))
    
