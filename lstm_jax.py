import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import optax

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    #return series[..., np.newaxis].astype(np.float32)
    return series

def get_dataloader(X,y,batch_size,key,axis=0):
    num_train=X.shape[axis]
    indices = jnp.array(list(range(0,num_train)))
    indices=jax.random.permutation(key,indices)
    for i in range(0, len(indices),batch_size):
        batch_indices = jnp.array(indices[i: i+batch_size])
        yield jnp.asarray(X[batch_indices,:]), jnp.asarray(y[batch_indices])


class LSTM(nn.Module):
    
    @nn.remat    
    @nn.compact   
    def __call__(self, X_batch):
        carry,x=nn.RNN(nn.LSTMCell(20),return_carry=True)(X_batch)
        carry,x=nn.RNN(nn.LSTMCell(20),return_carry=True)(x)
        x=nn.Dense(10)(x)
        return x
    

def log_likelihood(params, x, y):
    preds = model.apply(params, x)
    return jnp.mean(optax.l2_loss(y,preds).sum(axis=-1))

grad_log_post=jax.jit(jax.grad(log_likelihood))

def sgd(key,log_post, grad_log_post, num_samples,
                             dt, x_0,train_data,test_data,batch_size):
    samples = list()
    loss=list()
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
        if (i%(num_samples//10)==0):
            print('iteration {0}, loss {1:.2f}'.format(i,loss[-1]))
    return samples,loss


np.random.seed(42)

n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]

Y_train = series[:7000,n_steps:]
Y_valid = series[7000:9000,n_steps:]
Y_test = series[9000:,n_steps:]


batch_size = 32

key=jax.random.PRNGKey(32)
key_model,key_data=jax.random.split(key,2)
inputs = jax.random.randint(key,(batch_size, n_steps),0, 10,).astype(jnp.float32)
train_data=X_train,Y_train
test_data=X_test,Y_test
batch_size=32
num_samples=100
dt=1e-3
model=LSTM()
params=model.init(key,inputs)

params,loss=sgd(key_data,log_likelihood, grad_log_post, num_samples,dt,params,train_data,test_data,batch_size)