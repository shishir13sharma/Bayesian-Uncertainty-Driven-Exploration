import math
import torch
import numpy as np
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DifferentiableGaussian(nn.Module):    
    def __init__(self, mean, rho):
        super().__init__()
        self.mean = mean
        self.rho = rho
        self.gaussian = torch.distributions.Normal(0, 1)
        
    @property
    def sigma(self):
        return torch.log(1 + torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.gaussian.sample(self.rho.size()).to(DEVICE)
        return self.mean.to(DEVICE) + epsilon * self.sigma.to(DEVICE)
    
    def log_prob(self, obs):
        diff = self.mean.view(-1) - obs.view(-1)
        sigma = self.sigma.view(-1)
        precision = 1/self.sigma.view(-1)
        
        return -0.5 * (torch.log(2*np.pi*sigma*sigma) + torch.pow(diff*precision, 2)).sum()

class ScaleMixtureGaussian():    
    def __init__(self, pi, sigma1, sigma2):
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dist1 = torch.distributions.Normal(0, sigma1)
        self.dist2 = torch.distributions.Normal(0, sigma2)
        
        self.sigma1.requires_grad = False
        self.sigma2.requires_grad = False
        
    def log_prob(self, obs):
        prob1 = torch.exp(self.dist1.log_prob(obs))
        prob2 = torch.exp(self.dist2.log_prob(obs))
        return torch.log(self.pi*prob1 + (1 - self.pi)*prob2).sum()    