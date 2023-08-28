import pandas as pd
import matplotlib.pyplot as plt
import keras_core as keras
import os
os.environ["KERAS_BACKEND"] = 'jax'
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import optax

csv_path = 'data/jena_climate_2009_2016.csv'
df = pd.read_csv(csv_path)
split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 6

past = 720
future = 72
learning_rate = 0.001
batch_size = 256
epochs = 10

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

titles = [
    "Pressure",
    "Temperature",
    "Temperature in Kelvin",
    "Temperature (dew point)",
    "Relative Humidity",
    "Saturation vapor pressure",
    "Vapor pressure",
    "Vapor pressure deficit",
    "Specific humidity",
    "Water vapor concentration",
    "Airtight",
    "Wind speed",
    "Maximum wind speed",
    "Wind direction in degrees",
]

feature_keys = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]
date_time_key = "Date Time"
print(
    "The selected parameters are:",
    ", ".join([titles[i] for i in [0, 1, 5, 7, 8, 10, 11]]),
)

selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
features = df[selected_features]
features.index = df[date_time_key]
features.head()

features = normalize(features.values, train_split)
features = pd.DataFrame(features)
features.head()

train_data = features.loc[0 : train_split - 1]
val_data = features.loc[train_split:]

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(7)]].values
y_train = features.iloc[start:end][[1]]

sequence_length = int(past / step)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
y_val = features.iloc[label_start:][[1]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    X, y = batch

print("Input shape:", X.numpy().shape)
print("Target shape:", y.numpy().shape)

class LSTM(nn.Module):

    @nn.remat    
    @nn.compact   
    def __call__(self, X_batch):
        carry,x=nn.RNN(nn.LSTMCell(32),return_carry=True)(X_batch)
        x=nn.Dense(1)(x)
        return x[:,-1,:]
    
def log_likelihood(params, x, y):
    preds = model.apply(params, x)
    return jnp.mean(optax.l2_loss(y,preds).sum(axis=-1))

grad_log_post=jax.jit(jax.grad(log_likelihood))

def sgd(key,log_post, grad_log_post, num_samples,
                             dt, x_0,train_data,test_data=None):
    samples = list()
    loss=list()
    param = x_0
    optimizer = optax.sgd(dt,momentum=0.9,nesterov=True)
    opt_state = optimizer.init(param)
    for i in range(num_samples):
        for _,(X_batch, y_batch) in enumerate(train_data):
            X_batch=jnp.array(X_batch.numpy())
            y_batch=jnp.array(y_batch.numpy())
            grads = grad_log_post(param,X_batch,y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            param = optax.apply_updates(param, updates)
        loss.append(log_post(param,X_batch,y_batch))
        samples.append(param)
        if (i%(num_samples//10)==0):
            print('iteration {0}, loss {1:.2f}'.format(i,loss[-1]))
    return samples,loss

key=jax.random.PRNGKey(32)
key_model,key_data=jax.random.split(key,2)
inputs = jax.random.randint(key,(batch_size, sequence_length,7),0, 10,).astype(jnp.float32)
model=LSTM()
params=model.init(key,inputs)
num_samples=10
dt=1e-3
params,loss=sgd(key_data,log_likelihood, grad_log_post, num_samples,dt,params,dataset_train,dataset_val)