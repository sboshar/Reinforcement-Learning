import torch
class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.masks.clear()  
    
    # def get_all(self):
    #     return torch.cat(self.log_probs), torch.cat(self.returns).detach(), torch.cat(values)
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.masks)
    
    def __iter__(self):
        return self._zip() 
        # for data in self._zip():
            # return data
    
    # def reversed(self):
    #     for data in list(self._zip())[::-1]:
    #         yield data
    
    def __len__(self):
        return len(self.rewards)
    
        #maybe ill move this ot memory al together.
    def compute_returns(self, next_value, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(self))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns
    
    def compute_gae(self, next_value, gamma=0.99, lam=0.97):
        values = self.values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(self))):
            delta = self.rewards[step] + gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + gamma * lam * self.masks[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + values[step])
        return returns
    
    def test(self):
        print(len(self))

if __name__ == '__main__':
    m = Memory()
    m.add(1,2,3,4)
    m.add(1,5,3,4)
    m.add(1,4,3,4)
    m.test()

    # for i in m:
    #     print(i)
    
    # for i, data in enumerate(m.reversed()):
    #     print(i, data)
    