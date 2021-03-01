import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, partial, grad
from jax import nn, random
from jax import scipy
import tensorflow_datasets as tfds
import time
from util import one_hot, read_idx
"""
Load data:
1. MNIST
"""

# 1. MNIST
# ======
# load data
data_dir = '/tmp/tfds'

# Fetch full datasets for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)
data_train, data_test = mnist_data['train'], mnist_data['test']


# def one_hot(x, k, dtype=jnp.float32):
#     "Create a one-hot encoding of x of size k."
#     return jnp.array(x[:, None] == jnp.arange(k), dtype)

y_train = one_hot(data_train['label'], 10)
y_test = one_hot(data_test['label'], 10)

X_train = data_train['image']
X_train = X_train.reshape(X_train.shape[0], 28*28)

X_test = data_test['image']
X_test = X_test.reshape(X_test.shape[0], 28*28)

# Normalizing the RGB codes by dividing it to the max RGB value.
X_train = X_train/255
X_test = X_test/255
X_train = jnp.array(X_train)
y_train = jnp.array(y_train)
N_data = X_train.shape[0]



# 2. notMNIST dataset
# Download from http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html and
# convert to correct format using https://github.com/davidflanagan/notMNIST-to-MNIST
# Note the wrong filename in the readme (https://github.com/davidflanagan/notMNIST-to-MNIST/issues/1)
try:
    y_train_notMNIST = one_hot(read_idx(f"data/train-labels-idx1-ubyte"), 10)
    X_train_notMNIST = read_idx(f"data/train-images-idx3-ubyte")
    X_train_notMNIST = X_train_notMNIST.reshape(X_train_notMNIST.shape[0], 28*28)

    y_test_notMNIST = one_hot(read_idx(f"data/t10k-labels-idx1-ubyte"), 10)
    X_test_notMNIST = read_idx(f"data/t10k-images-idx3-ubyte")
    X_test_notMNIST = X_test_notMNIST.reshape(X_test_notMNIST.shape[0], 28*28)

    # normalise the RGB codes
    X_train_notMNIST = X_train_notMNIST/255
    X_test_notMNIST = X_test_notMNIST/255
    X_train_notMNIST = jnp.array(X_train_notMNIST)
    y_train_notMNIST = jnp.array(y_train_notMNIST)
    N_data_notMNIST = X_train_notMNIST.shape[0]
except:
    print("notMNIST data not found. Download from http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html and convert using https://github.com/davidflanagan/notMNIST-to-MNIST")
