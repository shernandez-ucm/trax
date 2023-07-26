from warnings import filterwarnings

import jax
from jax import vmap, jit
import jax.numpy as jnp
from jax.random import PRNGKey, split, normal
import jax.random as random
import numpy as np
import distrax
import tensorflow_probability.substrates.jax.distributions as tfd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from functools import partial
from flax import linen as nn
import optax

def rotate(X, deg):
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c, -s], [s, c]])
    X = X.dot(R)
    return np.asarray(X)

def get_dataloader(X,y,batch_size,key,axis=0):
    num_train=X.shape[axis]
    indices = jnp.array(list(range(0,num_train)))
    indices=jax.random.permutation(key,indices)
    for i in range(0, len(indices),batch_size):
        batch_indices = jnp.array(indices[i: i+batch_size])
        yield X[:,batch_indices,:], y[:,batch_indices]


n_groups = 18
n_grps_sq = int(np.sqrt(n_groups))
n_samples = 100
np.random.seed(31)

Xs, Ys = [], []
for i in range(n_groups):
    X, Y = make_moons(noise=0.3, n_samples=n_samples)
    X = scale(X)
    rotate_by = np.random.randn() * 90.0
    X = rotate(X, rotate_by)
    Xs.append(X)
    Ys.append(Y)
Xs = jnp.stack(Xs)
Ys = jnp.stack(Ys)
Xs_train = Xs[:, : n_samples // 2, :]
Xs_test = Xs[:, n_samples // 2 :, :]
Ys_train = Ys[:, : n_samples // 2]
Ys_test = Ys[:, n_samples // 2 :]

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(5)(x))
        x = nn.tanh(nn.Dense(5)(x))
        x = nn.Dense(1)(x)
        return x

model = MLP()

def reparameterize(params):
    mu = params['mu']
    sigma = jax.tree_map(lambda p : jnp.exp(p),params['log_std'])
    eps = params['eps']
    model_params = jax.tree_map(lambda m,e,s : m+e*s,mu,eps,sigma)
    return model_params

def log_likelihood(params, x, y):
    model_params=reparameterize(params)
    logits = jnp.squeeze(jax.vmap(model.apply,(0,0))(model_params, x))
    return distrax.Bernoulli(logits=logits).log_prob(y).sum()

def log_prior(params):
    def flatten(params):
        squared=jax.tree_map(lambda p: distrax.Normal(0.0,1.0).log_prob(p).sum(), params)
        flattten_squared=jnp.sum(jnp.stack(jax.tree_util.tree_leaves(squared['params'])))
        return flattten_squared
    squared_mu=flatten(params['mu'])
    squared_noise=flatten(params['eps'])
    squared_std=flatten(params['log_std'])
    return squared_std+squared_noise+squared_mu

def log_post(params,batch,labels):
    n_data=batch.shape[0]
    return -1./n_data*(log_prior(params) + log_likelihood(params,batch,labels))

grad_log_post=jax.jit(jax.grad(log_post))

@jit
def accuracy_cnn(params, X, y):
    target_class = y
    model_params=reparameterize(params)
    logits=model.apply(model_params, X).ravel()
    predicted_class = nn.sigmoid(logits)>0.5
    return predicted_class == target_class

full_batch_evaluate=jax.vmap(accuracy_cnn,in_axes=(0,0,0))


@partial(jit, static_argnums=(2,3))
def sgld_kernel(key, params, grad_log_post, dt, X, y_data):
    key, subkey = random.split(key, 2)
    grads = grad_log_post(params, X, y_data)
    noise=jax.tree_map(lambda p: random.normal(key=subkey,shape=p.shape), params)
    params=jax.tree_map(lambda p, g,n: p-0.5*dt*g+jnp.sqrt(dt)*n, params, grads,noise)
    return key, params

def sgld(key,log_post, grad_log_post, num_samples,
                             dt, x_0,train_data,test_data,batch_size):
    samples = list()
    loss=list()
    accuracy=list()
    param = x_0
    key_train, key_model = jax.random.split(key, 2)
    for i in range(num_samples):
        train_data_loader = get_dataloader(train_data[0],train_data[1],batch_size,key_train)
        for _,(X_batch, y_batch) in enumerate(train_data_loader):
            key_model, param = sgld_kernel(key_model, param, grad_log_post, dt, X_batch, y_batch)
        loss.append(log_post(param,X_batch,y_batch))
        samples.append(param)
        test_acc=jnp.mean(full_batch_evaluate(param,test_data[0],test_data[1]))
        accuracy.append(test_acc)
        if (i%(num_samples//10)==0):
            print('iteration {0}, loss {1:.2f}, accuracy {2:.2f}'.format(i,loss[-1],accuracy[-1]))
    return samples,loss,accuracy

def sgd(key,log_post, grad_log_post, num_samples,
                             dt, x_0,train_data,test_data,batch_size):
    samples = list()
    loss=list()
    accuracy=list()
    param = x_0
    optimizer = optax.sgd(dt,momentum=0.9,nesterov=True)
    opt_state = optimizer.init(param)
    for i in range(num_samples):
        train_data_loader = get_dataloader(train_data[0],train_data[1],batch_size,key)
        for _,(X_batch, y_batch) in enumerate(train_data_loader):
            grads = grad_log_post(param,X_batch,y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            param = optax.apply_updates(param, updates)
        loss.append(log_post(param,X_batch,y_batch))
        samples.append(param)
        test_acc=jnp.mean(full_batch_evaluate(param,test_data[0],test_data[1]))
        accuracy.append(test_acc)
        if (i%(num_samples//10)==0):
            print('iteration {0}, loss {1:.2f}, accuracy {2:.2f}'.format(i,loss[-1],accuracy[-1]))
    return samples,loss,accuracy

key = jax.random.PRNGKey(2)
batch = jnp.ones((n_samples, 2))
key_model_init, key_state_init = jax.random.split(key, 2)
key_params_mu,key_params_sigma,key_params_eps=jax.random.split(key_model_init,3)
key_tasks=jax.random.split(key_params_mu,n_groups)
params_mu = jax.vmap(model.init, (0, None))(key_tasks, batch)
params_noise=jax.tree_map(lambda p: distrax.Normal(0.0,1.0).sample(seed=key_params_eps,sample_shape=p.shape),params_mu)
params_sigma=jax.tree_map(lambda p: distrax.Normal(0.0,1.0).sample(seed=key_params_sigma,sample_shape=p.shape),params_mu)
params_tasks = {
        'mu': params_mu,
        'eps': params_noise,
        'log_std': params_sigma,
    }
batch_size = 10 
train_data=Xs_train,Ys_train
test_data=Xs_test,Ys_test

samples,loss,accuracy=sgd(key_state_init,log_post,
                            grad_log_post,5_000,1e-3,
                            params_tasks,train_data,
                            test_data,batch_size)

results=full_batch_evaluate(samples[-1],Xs_test,Ys_test)
print('Test Accuracy {0:.2f}'.format(jnp.mean(results)))