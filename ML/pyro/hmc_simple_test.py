import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.infer import MCMC, NUTS
import pandas as pd

pyro.set_rng_seed(10244)
assert pyro.__version__.startswith('1.3.0')

'''
s * fm ** a * Zn ** b * c / Vr ** d
Dnozzle: diameter of nozzle
fm: flow speed
Zn: nozzle height
Vr: printing velocity
Zratio: Zn / Dnozzle
'''
    # Utility function to print latent sites' quantile information.
def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

def model(x, dims, obserations):
    a_loc = pyro.param('a_loc', torch.tensor(1.),
                        constraint=constraints.positive)
    b_loc = pyro.param('b_loc', torch.tensor(2.),
                        constraint=constraints.positive)
    c_loc = pyro.param('c_loc', torch.tensor(2.),
                        constraint=constraints.positive)
    d_loc = pyro.param('d_loc', torch.tensor(2.),
                        constraint=constraints.positive)
    # sigma_loc = pyro.param('sigma_loc', torch.tensor(0.),
    #                     constraint=constraints.positive)
    a = pyro.sample("a", dist.Normal(a_loc, 1.))
    b = pyro.sample("b", dist.Normal(b_loc, 1.))
    c = pyro.sample("c", dist.Normal(c_loc, 1.))
    d = pyro.sample("d", dist.Normal(d_loc, 1.))

    # mean = obserations.mean()
    s, fm, Zn, Vr = x
    mean = s * fm ** a * Zn ** b * c/ Vr ** d
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    # sigma = torch.tensor(0.)
    # sigma = dist.Uniform(0., 5.).sample()    
    # sigma = dist.Uniform(sigma_loc, 5.).sample()
    # sigma = dist.Normal(sigma_loc, 0.2).sample()
    pyro.sample("obs", dist.Normal(mean, sigma), obs=obserations)

dims = 4
num_samples = 100

# generate observations
x = torch.rand(dims, num_samples)
noise = torch.distributions.Normal(torch.tensor([0.]*num_samples), torch.tensor([0.2]*num_samples)).rsample()
s, fm, Zn, Vr = x
a, b, c, d = 1.5, 1.8, 2.1, 2.3
# a, b, c, d = 1., 1., 1., 1.
obserations = s * fm ** a * Zn ** b * c/ Vr ** d + noise[0]

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=400)
mcmc.run(x, dims, obserations)

hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
for site, values in summary(hmc_samples).items():
    print("Site: {}".format(site))
    print(values, "\n")





