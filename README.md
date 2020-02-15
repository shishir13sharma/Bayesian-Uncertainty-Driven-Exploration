# Bayesian Uncertainty Driven Exploration

The project investigates the methodology of posterior sampling using Bayesian Networks for driving exploration in complex environments. Posterior sampling allows the agent to perform deep exploration of the environment by sampling different Q-value function for each episode. The requirement for such a strategy is to maintaina distribution over Q-value functions. The exploration in this strategy is driven by the variance of the posterior distribution that is sampled from each episode. 

The Bayes-By-Backprop algorithm (Blundell etal., 2015) is employed to maintain a Bayesian Network that acts as the distribution over Q-value functions and is efficiently updated using Backpropagation algorithm.

## Running

The 3 notebooks correspond to the 3 different environments i.e. Chain, CartPole and Pendulum. 
Simply running the notebooks should start training the network.

Tensorboard can be used to view the plots saved in results folder by 

```
tensorboard --logdir results/Cartpole.
```