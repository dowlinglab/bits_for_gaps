"""Small array helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf


def standardize(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Zero-mean, unit-variance standardization.

    Parameters
    ----------
    data : np.ndarray

    Returns
    -------
    z : np.ndarray
        ``(data - mean) / std``.
    mean, std : float
    """
    mean, std_dev = np.mean(data), np.std(data)
    z = (data - mean) / std_dev
    return z, mean, std_dev


def normalize(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Min-max normalization to [0, 1] (reversed: ``max`` maps to 0, ``min`` to 1).

    Parameters
    ----------
    data : np.ndarray

    Returns
    -------
    z : np.ndarray
        ``(max(data) - data) / (max(data) - min(data))``.
    max_data, min_data : float
    """
    max_data, min_data = max(data), min(data)
    z = (max_data - data) / (max_data - min_data)
    return z, max_data, min_data


def make_tensor(x: np.ndarray) -> tf.Tensor:
    """Convert a 1-D or 2-D NumPy array to a float64 TF tensor (column vector if 1-D).

    Parameters
    ----------
    x : np.ndarray, shape (n,) or (n, d)

    Returns
    -------
    tf.Tensor, shape (n, 1) if ``x`` was 1-D, else (n, d)
    """
    if x.ndim == 1:
        x_tensor = tf.convert_to_tensor(x[:, None], dtype=tf.float64)
    else:
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
    return x_tensor
