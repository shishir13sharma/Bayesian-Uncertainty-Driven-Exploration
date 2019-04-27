import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from Distributions import DifferentiableGaussian, ScaleMixtureGaussian

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BayesianLinear(nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        #Weight Parameters 
        self.init_weight_mean = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.init_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-2, 2))
        self.weight = DifferentiableGaussian(self.init_weight_mean, self.init_weight_rho)
        
        #Bias Parameters 
        self.init_bias_mean = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.init_bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-2, 2))
        self.bias = DifferentiableGaussian(self.init_bias_mean, self.init_bias_rho)
        
        #Prior Distributions
        pi = 0.5
        prior_sigma1 = torch.FloatTensor([math.exp(-0)])
        prior_sigma2 = torch.FloatTensor([math.exp(1)])
        self.weight_prior = ScaleMixtureGaussian(pi, prior_sigma1, prior_sigma2)
        self.bias_prior = ScaleMixtureGaussian(pi, prior_sigma1, prior_sigma2)
        
        self.weight_sample, self.bias_sample = self.weight.sample().detach(), self.bias.sample().detach()
        self.log_prior = 0
        self.log_variational_posterior = 0
    
    def forward(self, input, use_sample, num_sample=0):
        
        if torch.isnan(self.weight.mean).sum().item() == 1:
            pdb.set_trace()
            
        if use_sample:       
            weight = self.weight.sample()
            bias = self.bias.sample() 
            if not num_sample:
                weight.data = self.weight_sample.data
                bias.data = self.bias_sample.data
            else:
                self.weight_sample.data = weight.data
                self.bias_sample.data = bias.data

        else:
            weight = self.weight.mean
            bias = self.bias.mean
                    
        if self.training :
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        
        return F.linear(input, weight, bias)

class BayesianNetwork(nn.Module):
    
    def __init__(self, features_list, train_samples, batch_size, batch_scale):
        super().__init__()
        
        self.train_samples = train_samples
        layer_arr = []
        for i in range(len(features_list) - 1):
            layer_arr.append(BayesianLinear(features_list[i], features_list[i+1]))        
                
        self.layer_arr = nn.ModuleList(layer_arr)       
        self.layer_num = len(layer_arr)
                
        self.batch_size = batch_size
        self.batch_scale = batch_scale
        
    def forward(self, x, use_sample, num_sample=0):        
        for i in range(self.layer_num - 1):            
            x = F.relu(self.layer_arr[i](x, use_sample, num_sample))            
        x = self.layer_arr[i+1](x, use_sample, num_sample)
        return x
        
    def log_prior(self):
        log_prior = 0        
        for i in range(self.layer_num):
            log_prior = log_prior + self.layer_arr[i].log_prior
        return log_prior
    
    def log_variational_posterior(self):
        log_posterior = 0
        for i in range(self.layer_num):
            log_posterior = log_posterior + self.layer_arr[i].log_variational_posterior
        return log_posterior
            
    def sample_elbo(self, obs, act, targets):
        
        outputs = torch.zeros(self.train_samples, self.batch_size).to(DEVICE)
        log_prior = torch.zeros(self.train_samples).to(DEVICE)
        log_variational_posterior = torch.zeros(self.train_samples).to(DEVICE)
        for i in range(self.train_samples):
            val = self.forward(obs, use_sample=True, num_sample=1)
            outputs[i] = val.gather(1, act.view(-1, 1)).squeeze()
            log_prior[i] = log_prior[i] + self.log_prior()/self.batch_scale
            log_variational_posterior[i] = log_variational_posterior[i] + self.log_variational_posterior()/self.batch_scale
        
        loss = nn.MSELoss(reduction='none')
        mse = loss(outputs.mean(0), targets)
        log_prior = log_prior.mean()      
        log_variational_posterior = log_variational_posterior.mean()
        td_errors = outputs.mean(0) - targets
        
        return log_prior, log_variational_posterior, mse, td_errors.detach().cpu().numpy()

