#bulding NN model

import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Input, Dense, Dropout ,Conv1D
def model_fn(ly1_units=20,activation_1='sigmoid',activation_2='tanh',ly2_units=20,activation_3='tanh',\
             dropout_rate1 = 0.1,dropout_rate2 = 0.1,loss_fn="huber_loss", krnl_sz=5,\
            optimizer='adam'):
    inputs = Input(shape=(17,1))
    x = Dense(ly1_units, activation=activation_1)(inputs)
   # print(x.shape)
    x=  Conv1D(filters=1,kernel_size=krnl_sz)(x)
    x= Flatten()(x)
    x = Dropout(dropout_rate1)(x, training=True)
    x = Dense(ly2_units, activation=activation_2)(x)
    x = Dropout(dropout_rate2)(x, training=True)
    x = Dense(110, activation=activation_3)(x)
    outputs = x
    model = Model(inputs, outputs)
#model.compile(loss="mean_squared_error", optimizer='adam')
    model.compile(loss=loss_fn, optimizer=optimizer)
    #model.summary()
    return model


