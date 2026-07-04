"""Small array helpers. Moved from the paper code's ``fxns/util.py``."""

import numpy as np
import tensorflow as tf


def standardize(data):
    """Zero-mean, unit-variance standardization. Returns (z, mean, std)."""
    mean, std_dev = np.mean(data), np.std(data)
    z = (data - mean) / std_dev
    return z, mean, std_dev


def normalize(data):
    """Min-max normalization to [0, 1] (reversed). Returns (z, max, min)."""
    max_data, min_data = max(data), min(data)
    z = (max_data - data) / (max_data - min_data)
    return z, max_data, min_data


def make_tensor(x):
    """Convert a 1-D or 2-D NumPy array to a float64 TF tensor (column vector if 1-D)."""
    if x.ndim == 1:
        x_tensor = tf.convert_to_tensor(x[:, None], dtype=tf.float64)
    else:
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
    return x_tensor
