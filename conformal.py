import numpy as np
import matplotlib.pyplot as plt


def make_sinusoidal(n_data: int, noise: float = 0.1, seed: int = 0):
    np.random.seed(seed)
    w = np.arange(30)[:, None] / 30
    b = 2 * np.pi * np.arange(30)[:, None] / 30

    x = np.random.normal(size=(n_data,))
    y = np.cos(w * x + b).sum(0) + noise * np.random.normal(size=(n_data,))
    return x[:, None], y[:, None]


train_data = make_sinusoidal(n_data=500, seed=0)
val_data = make_sinusoidal(n_data=500, seed=1)
test_data = make_sinusoidal(n_data=500, seed=2)

fig, axes = plt.subplots(1, 3, figsize=(10, 1))
axes[0].scatter(*train_data, s=1, label="training data", c="C0")
axes[0].legend()
axes[1].scatter(*val_data, s=1, label="validation data", c="C1")
axes[1].legend()
axes[2].scatter(*test_data, s=1, label="test data", c="C2")
axes[2].legend()
#plt.show()

from fortuna.data import DataLoader

train_data_loader = DataLoader.from_array_data(
    train_data, batch_size=128, shuffle=True, prefetch=True
)
val_data_loader = DataLoader.from_array_data(val_data, batch_size=128, prefetch=True)
test_data_loader = DataLoader.from_array_data(test_data, batch_size=128, prefetch=True)