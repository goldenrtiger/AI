# ---------------------------------------------------------------------
# ---------------------------------------
# 
# python: 3.7
# tensorflow: 2.1.0
# date: 04/04/2020
# author: Xu Jing
# email: xj.yixing@hotmail.com
# 
# ---------------------------------------
# ----------------------------------------------------------------------

# ---------------------------------------------------------------------
# a = pyro.distributions.Normal(0.0, 1.) # create an object
# a.rsample()    #execute sample
# equivalent to the following code
# pyro.sample('test', a) 
# pyro.sample('test', a, obs=torch.tensor(1.))
# >>> out:
#   tensor(1.) # always
# ---------------------------------------------------------------------


import torch 
import pyro

# torch >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> -------- >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ---------------------------------------------------------------------
# tensor
# rsample, sample, rand
# 
# ---------------------------------------------------------------------


### --------------------------------------------------------------------------------------------------------------
# x = torch.tensor([[1]])
# print(f"toch.tensor([[1]]):{x}, x.item():{x.item()}")
# # toch.tensor([[1]]):tensor([[1]]), x.item():1

### --------------------------------------------------------------------------------------------------------------
# x = torch.tensor([[1,2,3],[4,5,6]])
# print(f"torch.tensor([[1,2,3],[4,5,6]]):{x}, x[1][1]:{x[1][1]}")
# # torch.tensor([[1,2,3],[4,5,6]]):tensor([[1, 2, 3],
# #         [4, 5, 6]]), x[1][1]:5


### --------------------------------------------------------------------------------------------------------------
# '''
# x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
# out = x.pow(2).sum()
# out.backward()
# grad = x.grad
# print(f"x.grad:{grad}")
# # x.grad:tensor([[ 2., -2.],
# #         [ 2.,  2.]])
# '''

### ---------------------------------------------------------------------------------------
# normal = torch.distributions.Normal(0., 1.)
# x = normal.rsample(torch.tensor([3,4]))
# print(f"x:{x}")
# # >>>out:
# # x:tensor([[-1.1206, -0.8669, -0.4834,  0.5253],
# #         [-0.2134,  0.1806, -0.4429, -0.8174],
# #         [-0.3579,  1.4537, -3.0391, -1.0096]])
# y = normal.sample(torch.tensor([3,4]))
# print(f"y:{y}")
# # >>>out:
# # y:tensor([[ 0.5678,  1.1513, -0.8438, -0.4082],
# #         [ 0.0012, -0.1341, -0.6118, -1.0525],
# #         [-0.3909, -0.0978,  0.3985, -1.2088]])
# z = torch.rand(3,4)
# print(f"z:{z}")
# # >>>out:
# # z:tensor([[0.8446, 0.3177, 0.8650, 0.0365],
# #         [0.9405, 0.0388, 0.1358, 0.8171],
# #         [0.2011, 0.4106, 0.4717, 0.0533]])



# pyro 1.3.0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> -------- >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ---------------------------------------------------------------------
# pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3), torch.tensor([3,4]))
# 
# ---------------------------------------------------------------------

### --------------------------------------------------------------------------------------------------------------
# pyro.set_rng_seed(101)
# loc = 0.    # mean
# scale = 1.  # scale
# normal = torch.distributions.Normal(loc, scale) # create an object
# x = normal.rsample() # draw a sample form N(0,1)

# print("sample:", x)
# print("log prob", normal.log_prob(x)) # score the sample from N(0,1)
# print("log prob", normal.log_prob(torch.tensor(-1.3))) # score the sample from N(0,1)

### --------------------------------------------------------------------------------------------------------------
# def weather():
#     cloudy = torch.distributions.Bernoulli(0.3).sample()
#     cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
#     mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
#     scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
#     temp = torch.distributions.Normal(mean_temp, scale_temp).rsample()
#     return cloudy, temp.item()

### --------------------------------------------------------------------------------------------------------------
# def weather():
#     cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
#     cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
#     mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
#     scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
#     temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp))
#     return cloudy, temp.item()

# for _ in range(3):
#     print(weather())

# def ice_cream_sales():
#     cloudy, temp = weather()
#     expected_sales = 200. if cloudy == 'sunny' and temp > 80.0 else 50.
#     ice_cream = pyro.sample('ice_cream', pyro.distributions.Normal(expected_sales, 10.0))
#     return ice_cream

### recursive --------------------------------------------------------------------------------------------------------------
# def geometric(p, t=None):
#     if t is None:
#         t = 0
#     x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(p))
#     if x.item() == 1:
#         print(f"0")
#         return 0
#     else:
#         print(f"continued")
#         return 1 + geometric(p, t + 1)

# print(geometric(1, 0.1))

### --------------------------------------------------------------------------------------------------------------
# def normal_product(loc, scale):
#     z1 = pyro.sample("z1", pyro.distributions.Normal(loc, scale))
#     z2 = pyro.sample("z2", pyro.distributions.Normal(loc, scale))
#     y = z1 * z2
#     return y

# def make_normal_normal():
#     mu_latent = pyro.sample("mu_latent", pyro.distributions.Normal(0, 1))
#     fn = lambda scale: normal_product(mu_latent, scale)
#     return fn

# print(make_normal_normal()(1.))


### condition --------------------------------------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# import pyro
# import pyro.infer
# import pyro.optim
# import pyro.distributions as dist

# pyro.set_rng_seed(101)

# # weight|guess ~ Normal(guess, 1)
# # measurement|guess, weight ~ Normal(weight, 0.75)
# def scale(guess):
#     weight = pyro.sample("weight", dist.Normal(guess, 1.0))
#     return pyro.sample("measurement", dist.Normal(weight, 0.75))

# conditioned_scale = pyro.condition(scale, data={"measurement": 9.5})

# def deferred_conditioned_scale(measurement, guess):
#     return pyro.condition(scale, data={"measurement": measurement})(guess)

# # ---------------------------------------------------
# def scale_obs(guess):  # equivalent to conditioned_scale above
#     weight = pyro.sample("weight", dist.Normal(guess, 1.))
#      # here we condition on measurement == 9.5
#     return pyro.sample("measurement", dist.Normal(weight, 0.75), obs=9.5)

# # ----------------------------------------------------
# # (for derivation, see for example Section 3.4 of 
# # http://www.stat.cmu.edu/~brian/463-663/week09/Chapter%2003.pdf ).
# # ----------------------------------------------------
# def perfect_guide(guess):
#     loc =(0.75**2 * guess + 9.5) / (1 + 0.75**2) # 9.14
#     scale = np.sqrt(0.75**2/(1 + 0.75**2)) # 0.6
#     return pyro.sample("weight", dist.Normal(loc, scale))

# def intractable_scale(guess):
#     weight = pyro.sample("weight", dist.Normal(guess, 1.0))
#     return pyro.sample("measurement", dist.Normal(some_nonlinear_function(weight), 0.75))

# simple_param_store = {}
# a = simple_param_store.setdefault("a", torch.randn(1))

# def scale_parametrized_guide(guess):
#     a = pyro.param("a", torch.tensor(guess))
#     b = pyro.param("b", torch.tensor(1.))
#     return pyro.sample("weight", dist.Normal(a, torch.abs(b)))

# from torch.distributions import constraints

# def scale_parametrized_guide_constrained(guess):
#     a = pyro.param("a", torch.tensor(guess))
#     b = pyro.param("b", torch.tensor(1.), constraint=constraints.positive)
#     return pyro.sample("weight", dist.Normal(a, b))  # no more torch.abs

# guess = 8.5

# pyro.clear_param_store()
# svi = pyro.infer.SVI(model=conditioned_scale,
#                      guide=scale_parametrized_guide,
#                      optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
#                      loss=pyro.infer.Trace_ELBO())

# losses, a, b = [], [], []
# num_steps = 2500
# for t in range(num_steps):
#     losses.append(svi.step(guess))
#     a.append(pyro.param("a").item())
#     b.append(pyro.param("b").item())

# plt.plot(losses)
# plt.title("ELBO")
# plt.xlabel("step")
# plt.ylabel("loss");
# print('a = ',pyro.param("a").item())
# print('b = ', pyro.param("b").item())
# plt.show()

# plt.subplot(1,2,1)
# plt.plot([0,num_steps],[9.14,9.14], 'k:')
# plt.plot(a)
# plt.ylabel('a')

# plt.subplot(1,2,2)
# plt.ylabel('b')
# plt.plot([0,num_steps],[0.6,0.6], 'k:')
# plt.plot(b)
# plt.tight_layout()

# plt.show()

### --------------------------------------------------------------------------------------------------------------
# import math
# import os
# import torch
# import torch.distributions.constraints as constraints
# import pyro
# from pyro.optim import Adam
# from pyro.infer import SVI, Trace_ELBO
# import pyro.distributions as dist

# # this is for running the notebook in our testing framework
# smoke_test = ('CI' in os.environ)
# n_steps = 2 if smoke_test else 2000

# # enable validation (e.g. validate parameters of distributions)
# assert pyro.__version__.startswith('1.3.0')
# pyro.enable_validation(True)

# # clear the param store in case we're in a REPL
# pyro.clear_param_store()

# # create some data with 6 observed heads and 4 observed tails
# data = []
# for _ in range(6):
#     data.append(torch.tensor(1.0))
# for _ in range(4):
#     data.append(torch.tensor(0.0))

# def model(data):
#     # define the hyperparameters that control the beta prior
#     alpha0 = torch.tensor(10.0)
#     beta0 = torch.tensor(10.0)
#     # sample f from the beta prior
#     f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
#     # loop over the observed data
#     for i in range(len(data)):
#         # observe datapoint i using the bernoulli likelihood
#         pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])

# def guide(data):
#     # register the two variational parameters with Pyro
#     # - both parameters will have initial value 15.0.
#     # - because we invoke constraints.positive, the optimizer
#     # will take gradients on the unconstrained parameters
#     # (which are related to the constrained parameters by a log)
#     alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
#                          constraint=constraints.positive)
#     beta_q = pyro.param("beta_q", torch.tensor(15.0),
#                         constraint=constraints.positive)
#     # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
#     pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

# # setup the optimizer
# adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
# optimizer = Adam(adam_params)

# # setup the inference algorithm
# svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# # do gradient steps
# for step in range(n_steps):
#     svi.step(data)
#     if step % 100 == 0:
#         print('.', end='')

# # grab the learned variational parameters
# alpha_q = pyro.param("alpha_q").item()
# beta_q = pyro.param("beta_q").item()

# # here we use some facts about the beta distribution
# # compute the inferred mean of the coin's fairness
# inferred_mean = alpha_q / (alpha_q + beta_q)
# # compute inferred standard deviation
# factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
# inferred_std = inferred_mean * math.sqrt(factor)

# print("\nbased on the data and our prior belief, the fairness " +
#       "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))

### MCMC ----------------------------------------------
# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

# import argparse
# import logging

# import torch

# import data
# import pyro
# import pyro.distributions as dist
# import pyro.poutine as poutine
# from pyro.infer import MCMC, NUTS

# logging.basicConfig(format='%(message)s', level=logging.INFO)
# pyro.enable_validation(__debug__)
# pyro.set_rng_seed(0)

# def model(sigma):
#     eta = pyro.sample('eta', dist.Normal(torch.zeros(data.J), torch.ones(data.J)))
#     mu = pyro.sample('mu', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
#     tau = pyro.sample('tau', dist.HalfCauchy(scale=25 * torch.ones(1)))

#     theta = mu + tau * eta

#     return pyro.sample("obs", dist.Normal(theta, sigma))


# def conditioned_model(model, sigma, y):
#     return poutine.condition(model, data={"obs": y})(sigma)


# def main(args):
#     nuts_kernel = NUTS(conditioned_model, jit_compile=args.jit)
#     mcmc = MCMC(nuts_kernel,
#                 num_samples=args.num_samples,
#                 warmup_steps=args.warmup_steps,
#                 num_chains=args.num_chains)
#     mcmc.run(model, data.sigma, data.y)
#     mcmc.summary(prob=0.5)


# if __name__ == '__main__':
#     assert pyro.__version__.startswith('1.3.0')
#     parser = argparse.ArgumentParser(description='Eight Schools MCMC')
#     parser.add_argument('--num-samples', type=int, default=1000,
#                         help='number of MCMC samples (default: 1000)')
#     parser.add_argument('--num-chains', type=int, default=1,
#                         help='number of parallel MCMC chains (default: 1)')
#     parser.add_argument('--warmup-steps', type=int, default=1000,
#                         help='number of MCMC samples for warmup (default: 1000)')
#     parser.add_argument('--jit', action='store_true', default=False)
#     args = parser.parse_args()

#     main(args)
### ----------------------------------------------

### ----------------------------------------------

### ----------------------------------------------

### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------
### ----------------------------------------------








