
import numpy as np

class chain_env():
    
    def __init__(self, N):        
        self.N = N
        self.current_state = 1
        self.timer = 0
        self.action_space = [-1, 1]
        
    def step(self, act):
        assert act in self.action_space
        assert self.timer < self.N + 9

        done = 0
        rew = self.reward(act)
        info  = []
        
        self.timer = self.timer + 1        
        self.current_state = self.current_state + act
        
        if self.current_state >= self.N:
            self.current_state = self.N - 1
        if self.current_state < 0:
            self.current_state = 0
            
        if self.timer == self.N + 9:
            done = 1
            
        return self.therm_encoding(self.current_state), rew, done, info            
        
            
    def reward(self, act):        
        if self.current_state == self.N - 1 and act == 1:
            return 1
        if self.current_state == 0 and act == -1:
            return 1e-3
        return 0
        
    def reset(self):
        self.current_state = 1
        self.timer = 0
        return self.therm_encoding(self.current_state)
        
    def therm_encoding(self, x):
    
        state = np.zeros(self.N)
        state[:x + 1] = 1
        return state

if __name__ == '__main__':
    
    env = chain_env(20)
    done = False
    obs = env.reset()
    act = 2*np.random.binomial(1, 0.5) - 1

    episode_rew = 0
    episode_count = 0

    for i in range(10000):
        if done:

            print("Episode " + str(episode_count) + " reward = " + str(episode_rew))
            episode_rew = 0
            episode_count = episode_count + 1

            obs = env.reset()
            act = 2*np.random.binomial(1, 0.5) - 1

        obs, rew, done = env.step(act)    
        act = 2*np.random.binomial(1, 0.5) - 1

        episode_rew = episode_rew + rew

