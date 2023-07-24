import jax.numpy as jnp
from jax import random
import sys
sys.path.append('../SGMCMCjax')
from sgmcmcjax.samplers import build_sgld_sampler


# define model in JAX
def loglikelihood(theta, x):
    return -0.5*jnp.dot(x-theta, x-theta)

def logprior(theta):
    return -0.5*jnp.dot(theta, theta)*0.01

# generate dataset
N, D = 10_000, 100
key = random.PRNGKey(0)
X_data = random.normal(key, shape=(N, D))

# build sampler
batch_size = int(0.1*N)
dt = 1e-5
my_sampler = build_sgld_sampler(dt, loglikelihood, logprior, (X_data,), 
batch_size)

# run sampler
Nsamples = 10_000
samples = my_sampler(key, Nsamples, jnp.zeros(D))
