import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
from tqdm.auto import tqdm


import mxnet as mx
from flax import linen as nn


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x

cnn = CNN()

def get_mx_datasets(mini_batch=32):
    def transform(data, label):
        data = data.astype('float32')/255
        return data.asnumpy(), label.asnumpy()
    train_dataset = mx.gluon.data.vision.datasets.MNIST(train=True).transform(transform)
    valid_dataset = mx.gluon.data.vision.datasets.MNIST(train=False).transform(transform)
    train_data_loader = mx.gluon.data.DataLoader(train_dataset, batch_size=mini_batch, num_workers=0)
    test_data_loader = mx.gluon.data.DataLoader(valid_dataset, batch_size=mini_batch, num_workers=0)
    return train_data_loader,test_data_loader

def log_likelihood(params, x, y):
    logits = cnn.apply(params, x)
    label = jax.nn.one_hot(y, num_classes=10)
    return jnp.sum(logits*label)

def log_prior(params):
    squared_params=jax.tree_map(lambda p: -0.5*jnp.sum(p*p), params)
    return jnp.sum(jnp.stack(jax.tree_util.tree_leaves(squared_params['params'])))

def log_post(params,batch,labels):
    n_data=batch.shape[0]
    return log_prior(params) + log_likelihood(params,batch,labels)

grad_log_post=jax.jit(jax.grad(log_post))

@jit
def accuracy_cnn(params, X, y):
    target_class = y
    predicted_class = jnp.argmax(cnn.apply(params, X),axis=1)
    return predicted_class == target_class

def evaluate(params,test_data_loader):
    acc=list()
    for i,(X_batch, y_batch) in tqdm(enumerate(test_data_loader)):
        X_batch=jnp.asarray(X_batch.asnumpy())
        y_batch=jnp.asarray(y_batch.asnumpy())
        accuracy_batch=accuracy_cnn(params,X_batch, y_batch)
        acc.append(accuracy_batch)
    return jnp.mean(jnp.concatenate(acc))

@partial(jit, static_argnums=(2,3))
def sgld_kernel(key, params, grad_log_post, dt, X, y_data):
    N = X.shape[0]
    key, subkey1, subkey2 = random.split(key, 3)
    #idx_batch = random.choice(subkey1, N, shape=(minibatch_size,))
    grads = grad_log_post(params, X, y_data)
    noise=jax.tree_map(lambda p: random.normal(key=subkey2,shape=p.shape), params)
    params=jax.tree_map(lambda p, g,n: p+dt*g+jnp.sqrt(2*dt) * n, params, grads,noise)
    return key, params

def sgld_sampler_jax_kernel(key,log_post, grad_log_post, num_samples,
                             dt, x_0,train_data_loader,test_data_loader):
    samples = list()
    loss=list()
    accuracy=list()
    param = x_0
    for i in range(num_samples):
        for _,(X_batch, y_batch) in enumerate(train_data_loader):
            X_batch=jnp.asarray(X_batch.asnumpy())
            y_batch=jnp.asarray(y_batch.asnumpy())
            key, param = sgld_kernel(key, param, grad_log_post, dt, X_batch, y_batch)
        loss.append(log_post(param,X_batch,y_batch))
        samples.append(param)
        accuracy.append(evaluate(param,test_data_loader))
        if (i%(num_samples//10)==0):
            print('iteration {}, loss {}, accuracy {}'.format(i,loss[-1],accuracy[-1]))
    return samples,loss,accuracy

@partial(jit, static_argnums=(1,2,3))
def sgld_sampler_full_jax(key, grad_log_post, num_samples, dt, x_0, train_data_loader,test_data_loader):
    
    def sgld_step(carry, x):
        key, param = carry
        key, param = sgld_kernel(key, param, grad_log_post, dt, train_data_loader,test_data_loader)
        return (key, param), param
    carry = (key, x_0)
    _, samples = jax.lax.scan(sgld_step, carry, None, num_samples)
    return samples
    
train_ds, test_ds = get_mx_datasets()

batch_size = 32
key=jax.random.PRNGKey(32)
param_key,sampler_key=jax.random.split(key)
params=cnn.init(param_key,jnp.ones([1,28,28,1]))
samples,loss,accuracy=sgld_sampler_jax_kernel(sampler_key,log_post,
                                               grad_log_post,100,1e-5,
                                               params,train_ds,test_ds)
