import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Input, Dense, Dropout ,Conv1D

import model_fn
estimator=tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=model_fn.model_fn)
