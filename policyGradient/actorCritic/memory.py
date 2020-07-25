class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        return self._zip() 
        # for data in self._zip():
            # return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

if __name__ == '__main__':
    m = Memory()
    m.add(1,2,3,4)
    m.add(1,5,3,4)
    m.add(1,4,3,4)
    for i in m:
        print(i)
    
    for i, data in enumerate(m.reversed()):
        print(i, data)
    