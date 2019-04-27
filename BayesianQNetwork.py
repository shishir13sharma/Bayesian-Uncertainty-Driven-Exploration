import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from baselines.deepq import ReplayBuffer, PrioritizedReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BQN_learn():
    
    def __init__(self, model, target_model, gamma, lr, batch_size, buffer_size, writer, train_start=100):
        super().__init__()
        
        self.dbqn = model
        self.target_dbqn = target_model
        self.update_target()
        
        self.t = 0
        self.gamma = gamma
        self.optimizer = optim.Adam(lr = lr, params = self.dbqn.parameters())
        self.batch_size = batch_size
        self.writer = writer
        
        self.train_freq = 1
        self.update_freq = 100
        self.train_start = train_start
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, 0.7)        
        
    def train(self):
        
        buffer_sample = self.replay_buffer.sample(self.batch_size, beta=0.5)
        obs = torch.from_numpy(buffer_sample[0]).float().to(DEVICE)
        act = torch.from_numpy(buffer_sample[1]).long().to(DEVICE)
        rew = torch.from_numpy(buffer_sample[2]).float().to(DEVICE)
        obs1 = torch.from_numpy(buffer_sample[3]).float().to(DEVICE)        
        dones = torch.from_numpy(buffer_sample[4].astype(int)).float().to(DEVICE)    
        wt = torch.from_numpy(buffer_sample[5]).float().to(DEVICE)    
        idxes = buffer_sample[6]
        
        self.dbqn.train()
        
        val1 = self.target_dbqn(obs1, use_sample=False).detach()
        _, max_act = val1.max(1)
        val1 = val1.gather(1, max_act.view(-1, 1)).squeeze()        
        targets = rew + self.gamma * val1 * (1 - dones)     
        
        log_prior, log_variational_posterior, mse, td_errors = self.dbqn.sample_elbo(obs, act, targets)                
        loss = (log_variational_posterior - log_prior) + mse        
                
        weighted_loss = (wt * loss).mean()        
        self.optimizer.zero_grad()
                
        weighted_loss.backward()        
        self.optimizer.step()
                        
        self.writer.add_scalar('data/loss', weighted_loss.detach().numpy(), self.t)        
        self.writer.add_scalar('data/prior', log_prior.detach().cpu().numpy(), self.t)
        self.writer.add_scalar('data/posterior', log_variational_posterior.detach().cpu().numpy(), self.t)
        self.writer.add_scalar('data/mse', mse.mean().detach().cpu().numpy(), self.t)        
                        
        self.writer.add_scalar('data/w1_sigma', np.mean(self.dbqn.layer_arr[0].weight.sigma[0].detach().cpu().numpy()).item(), self.t)              
        self.writer.add_scalar('data/w2_sigma', np.mean(self.dbqn.layer_arr[1].weight.sigma[0].detach().cpu().numpy()).item(), self.t)        
                
        return idxes, td_errors               
        
    def step(self, obs_t, act_t, rew_t, obs_t1, done ):
        
        self.replay_buffer.add(obs_t, act_t, rew_t, obs_t1, done)
        self.t = self.t + 1
        
        if self.t%self.train_freq == 0 and self.t > self.train_start:
            idxes, td_errors = self.train()
            self.replay_buffer.update_priorities(idxes, np.abs(td_errors) + 1e-6)
            
        if self.t%self.update_freq == 0 and self.t > self.train_start:
            self.update_target()
        
        return self.act(obs_t1, use_sample=True, num_sample=0)
        
    def act(self, obs, use_sample, num_sample):
        
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)   
        return np.argmax(self.dbqn(obs, use_sample, num_sample).detach().cpu().numpy())
        
    def reset(self, obs):        
        self.t = self.t + 1
        return self.act(obs, use_sample=True, num_sample=4)
        
    def update_target(self):
        self.target_dbqn.load_state_dict(self.dbqn.state_dict())        
        

