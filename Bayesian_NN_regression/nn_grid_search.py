import pickle
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

import seaborn as sns
sns.set()
from pandas.plotting import scatter_matrix
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from calculations_load import *
from configurations import *

# Get all the observables list

nobs        =  0
observables =  []
obs_name    =  []

for obs, cent_list in obs_cent_list['Pb-Pb-2760'].items():
    if obs not in active_obs_list['Pb-Pb-2760']:
        continue
    observables.append(obs)
    n = np.array(cent_list).shape[0]
    for i in cent_list:
        obs_name.append(f'{obs}_{i}')
    #self._slices[obs] = slice(self.nobs, self.nobs + n)
    nobs += n

system_str = 'Pb-Pb-2760'
design_file = 'production_designs/500pts/design_pts_Pb_Pb_2760_production/design_points_main_PbPb-2760.dat'
design = pd.read_csv(design_file)
design = design.drop("idx", axis=1)

#delete bad design points
drop_indices = list(delete_design_pts_set)
design = design.drop(drop_indices)

#choose features (inputs)
#feature_cols = ['norm', 'trento_p'] #specific choices
feature_cols = design.keys().values #all of them
n_features = len(feature_cols)

X = design[feature_cols]

n_design = SystemsInfo["Pb-Pb-2760"]["n_design"]
npt = n_design - len(delete_design_pts_set)
obs = 'dNch_deta' #choose the observable we want to emulate

Y = np.array([])

for pt in range(npt):
    for obs in active_obs_list['Pb-Pb-2760']:
        Y = np.append( Y, trimmed_model_data[system_str][pt, idf][obs]['mean'][:], axis=0)
Y = Y.reshape(X.shape[0], -1)


print( "X.shape : "+ str(X.shape) )
print( "Y.shape : "+ str(Y.shape) )

#Scaling the inputs

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

#X_scaler = StandardScaler().fit(X_train)
#Y_scaler = StandardScaler().fit(Y_train)
X_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
Y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(Y_train)

X_train_sc = X_scaler.transform(X_train)
X_test_sc = X_scaler.transform(X_test)

Y_train_sc = Y_scaler.transform(Y_train)
Y_test_sc = Y_scaler.transform(Y_test)


#Building NN model

from keras.models import Model
from keras.layers import Flatten, Input, Dense, Dropout ,Conv1D
def model_fn(ly1_units=20,activation_1='sigmoid',activation_2='tanh',ly2_units=20,activation_3='tanh',\
             dropout_rate1 = 0.1,dropout_rate2 = 0.1,loss_fn="huber_loss", krnl_sz=5,\
            optimizer='adam'):
    inputs = Input(shape=(X.shape[1],1))
    x = Dense(ly1_units, activation=activation_1)(inputs)
   # print(x.shape)
    x=  Conv1D(filters=1,kernel_size=krnl_sz)(x)
    x= Flatten()(x)
    x = Dropout(dropout_rate1)(x, training=True)
    x = Dense(ly2_units, activation=activation_2)(x)
    x = Dropout(dropout_rate2)(x, training=True)
    x = Dense(Y.shape[1], activation=activation_3)(x)
    outputs = x
    model = Model(inputs, outputs)
#model.compile(loss="mean_squared_error", optimizer='adam')
    model.compile(loss=loss_fn, optimizer=optimizer)
    model.summary()
    return model


#initiate models

model=model_fn()

#reshape inputs

train_tf_X=np.expand_dims(X_train_sc,axis=2)

#Grid search

from sklearn.model_selection import GridSearchCV
ly_units=[50,100,200,500]
activation_1=['sigmoid','tanh']
activation_2=['linear','tanh']
dropout_rate = [0.2, 0.3, 0.5]
krnl_sz=[5,10,20,40]
loss_fn=["mse","huber_loss"]
optimizer=['adam']
batch_size=[10, 20, 50]
estimator=tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=model_fn)
param_grid = dict(ly1_units=ly_units,ly2_units=ly_units,activation_1=activation_1, activation_2=activation_1,\
                activation_3=activation_2,dropout_rate1=dropout_rate,dropout_rate2=dropout_rate,\
                  loss_fn=loss_fn,optimizer=optimizer,batch_size=batch_size,krnl_sz=krnl_sz)
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=2, scoring='r2',verbose=20)
grid_result = grid.fit(train_tf_X, Y_train_sc,epochs=300,verbose=0)

print(f'The best set of hyperparameters are{grid_result.best_params_}')

file = open('grid_search_results.pkl', 'wb')

pickle.dump(grid_result, file)
