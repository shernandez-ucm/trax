import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import distrax
from functools import partial
from jax.scipy import stats
from jax.scipy.special import logsumexp

def get_dataloader(X,batch_size,key,axis=0):
    num_train=X.shape[axis]
    indices = jnp.array(list(range(0,num_train)))
    indices=jax.random.permutation(key,indices)
    for i in range(0, len(indices),batch_size):
        batch_indices = jnp.array(indices[i: i+batch_size])
        yield X[batch_indices]

def loglike_one_component(weight, mu, scale, data):
    return jnp.log(weight) + distrax.Normal(mu,scale).log_prob(data)

def loglike_across_components(weights, mus, scales, data):
    loglike_components = jax.vmap(partial(loglike_one_component,data=data))(
        weights, mus,scales
    )
    return logsumexp(loglike_components)

def log_likelihood(params, data):
    params['weights']=jax.nn.softmax(params['weights'])
    params['sigmas']=jax.vmap(jax.nn.softplus)(params['sigmas'])
    ll_per_data = jax.vmap(partial(loglike_across_components,params['weights'],params['mus'],params['sigmas']))(data)
    return -1.0*jnp.sum(ll_per_data)

def log_prior(params):
    log_prior=jax.tree.map(lambda p: distrax.Normal(0.0,1.0).log_prob(p), params['mus']).sum()
    log_prior+=distrax.Dirichlet(jnp.ones(3)).log_prob(params['weights'])
    return log_prior

def log_post(params,batch):
    n_data=batch.shape[0]
    return -1./n_data*log_prior(params) + log_likelihood(params,batch)

grad_log_post=jax.jit(jax.grad(log_post))

@partial(jax.jit, static_argnums=(3,4))
def sgld_kernel_momemtum(key, params, momemtum,grad_log_post, dt,batch):
    gamma,eps=0.9,1e-6
    key, subkey = jax.random.split(key, 2)
    grads = grad_log_post(params, batch)
    norm_grads=jax.tree.map(lambda g: g/jnp.linalg.norm(g),grads)
    squared_grads=jax.tree.map(lambda g: jnp.square(g),grads)
    momemtum=jax.tree.map(lambda m,s : gamma*m+(1-gamma)*s,momemtum,squared_grads)
    noise=jax.tree.map(lambda p: jax.random.normal(key=subkey,shape=p.shape), params)
    params=jax.tree.map(lambda p, g,m,n: p-0.5*dt*g/(m+eps)+jnp.sqrt(dt)*n, params, grads,momemtum,noise)
    return key, params,momemtum

def sgld(key,log_post, grad_log_post, num_samples,
                             dt, x_0,X,batch_size):
    samples = list()
    loss=list()
    param = x_0
    key_data, key_model = jax.random.split(key, 2)
    schedule_fn = lambda k: dt * k ** (-0.55)
    momemtum=jax.tree_map(lambda p:jnp.zeros_like(p),param)
    key_data_batch=jax.random.split(key_data, num_samples)
    for i in range(1,num_samples+1):
        train_data=get_dataloader(X,batch_size,key_data_batch[i])
        for _,X_batch in enumerate(train_data):
            key_model,param,momemtum = sgld_kernel_momemtum(key_model, param,momemtum, grad_log_post, schedule_fn(i), X_batch)
        loss.append(log_post(param,X_batch))
        samples.append(param)
        if (i%(num_samples//10)==0):
            print('iteration {0}, loss {1:.2f}'.format(i,loss[-1]))
    return samples,loss



RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

k = 3
ndata = 500
centers = np.array([-5.0, 0.0, 5.0])
sds = np.array([0.5, 2.0, 0.75])
idx = rng.integers(0, k, ndata)
X = rng.normal(loc=centers[idx], scale=sds[idx], size=ndata)

key=jax.random.PRNGKey(RANDOM_SEED)
key_model,key_data=jax.random.split(key,2)
key_mu,key_sigma=jax.random.split(key_model,2)
params={'weights':jnp.ones(3)/3.,
        'mus':jax.random.normal(key_mu,shape=(3,)),
        'sigmas':jax.random.normal(key_sigma,shape=(3,))}
samples,loss=sgld(key_data,log_post, grad_log_post, 5_000,1e-3,params,X,32)

posterior_means=jnp.stack([s['mus'] for s in samples])
posterior_weights=jnp.stack([s['weights'] for s in samples])
posterior_sigmas=jnp.stack([s['sigmas'] for s in samples])
