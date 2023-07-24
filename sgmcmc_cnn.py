import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax
from jax import random, jit

from tqdm.auto import tqdm

import sys
sys.path.append('../SGMCMCjax')
from sgmcmcjax.kernels import build_sgld_kernel, build_psgld_kernel,build_sgldAdam_kernel, build_sghmc_kernel

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

def loglikelihood(params, x, y):
    x = x[jnp.newaxis] # add an axis so that it works for a single data point
    logits = cnn.apply({'params':(params)}, x)
    label = jax.nn.one_hot(y, num_classes=10)
    return jnp.sum(logits*label)

def logprior(params):
    return 1.

def run_sgmcmc(key, Nsamples, init_fn, my_kernel, get_params, 
accuracy_rate=100):
    "Run SGMCMC sampler and return the test accuracy list"
    accuracy_list = []
    params = cnn.init(key, jnp.ones([1,28,28,1]))['params']
    key, subkey = random.split(key)
    state = init_fn(subkey, params)

    for i in tqdm(range(Nsamples)):
        key, subkey = random.split(key)
        state = my_kernel(i, subkey, state)
        if i%accuracy_rate==0:
            accuracy_list.append(accuracy_cnn(get_params(state), X_test_s, 
y_test_s))
    return accuracy_list

@jit
def accuracy_cnn(params, X, y):
    target_class = y
    predicted_class = jnp.argmax(cnn.apply({'params':(params)}, X), 
axis=1)
    return jnp.mean(predicted_class == target_class)


def get_mx_datasets():
    def transform(data, label):
        data = data.astype('float32')/255
        return data, label
    train_dataset = mx.gluon.data.vision.datasets.MNIST(train=True).transform(transform)
    valid_dataset = mx.gluon.data.vision.datasets.MNIST(train=False).transform(transform)
    n_train=len(train_dataset)
    n_valid=len(valid_dataset)
    return train_dataset.take(n_train)[:],valid_dataset.take(n_valid)[:]
    
train_ds, test_ds = get_mx_datasets()

X_train_s,y_train_s = train_ds
X_train_s,y_train_s=jnp.asarray(X_train_s.asnumpy()),jnp.asarray(y_train_s.asnumpy())
X_test_s,y_test_s = test_ds
X_test_s,y_test_s=jnp.asarray(X_test_s.asnumpy()),jnp.asarray(y_test_s.asnumpy())

data = (X_train_s, y_train_s)
batch_size = int(0.01*data[0].shape[0])

init_fn, sgld_kernel, get_params = build_sgld_kernel(5e-6, loglikelihood,logprior, data, batch_size)

sgld_kernel = jit(sgld_kernel)

Nsamples = 2000
accuracy_list_sgld = run_sgmcmc(random.PRNGKey(0), Nsamples, init_fn, 
sgld_kernel, get_params)
