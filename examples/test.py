import numpy as np
import tensorflow as tf
from scipy import sparse

tfla, tfr, tfs, tfm = tf.linalg, tf.random, tf.sparse, tf.math

tfla.cholesky_solve(tf.eye(5,5,[11], dtype=tf.float64),tf.eye(5,5,[11], dtype=tf.float64))
