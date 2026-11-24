#!/bin/python

####################
import os, sys

seed  = 120
train_size = 200  # number of events to train on
dir_tostore = 'pickled_mu2016'
gpu = 1
file_suffix = f'_{train_size}ev'  # used in filenames
####################

#################### Import things
import numpy as np
np.random.seed(seed=int(seed))
import pandas as pd
import pickle

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
sys.path.insert(0, '/eos/lhcb/user/h/havva/q2Measurement-BtoDstMuNu/GaussianProcessRegression/Packages/AmpliTF/')
sys.path.insert(0, "/eos/user/h/havva/.local/lib/python3.11/site-packages")

import gpflow
from gpflow.utilities import print_summary
from gpflow.config import default_float

from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set(font_scale=0.9) 
sns.set(style="ticks", color_codes=True)
mpl.rcParams.update({'font.size': 12.5})
plt.rcParams['axes.unicode_minus'] = False

from util_resolution import return_phasespace, Plot, MSE, return_quadsolution, get4mom, return_boostapprox, get_fitvars
from GPFlow_Model import Model
import amplitf.kinematics as atfk
####################

training_vars = ['Dst_PX', 'Dst_PY', 'Dst_PZ', 'Mu_PX', 'Mu_PY', 'Mu_PZ', 'B_Flight_X', 'B_Flight_Y', 'B_Flight_Z']
target_vars   = ['B_P_TRUE']

print(f'Reading ./forresolution/mu_{year}.p')
df = pd.read_pickle(f'./forresolution/mu_{year}.p')

#Drop rows with NaNs first
df = df.dropna(subset=training_vars + target_vars)

# Randomly sample 200 events
if train_size < len(df):
    df = df.sample(n=train_size, random_state=seed)
print(f'Shape of training df: {df.shape}')

# Define masses
mB = 5279.65000000
mDst = 2010.27000000 
ml  = 105.65837121
mTau= 1776.81991209 

#### TRAINING
X_train  = df[training_vars].values
Y_train  = df[target_vars].values

# Fit scalers on the training sample
scalerX.fit(X_train)
scalerY.fit(Y_train)

X_train_scaled = scalerX.transform(X_train)
Y_train_scaled = scalerY.transform(Y_train)

print('trainX', X_train_scaled)
print('trainY', Y_train_scaled)
print('Shape trainX', X_train_scaled.shape)
print('Shape trainY', Y_train_scaled.shape)

m = Model(X_train_scaled, Y_train_scaled, seed)

# Optimize using scipy
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=15000, disp=True))
print('Final model:', m)

# Predict on training sample
mu_train, var_train = m.predict_y(X_train_scaled)
mu_train = scalerY.inverse_transform(mu_train)
X_train_orig = scalerX.inverse_transform(X_train_scaled)
Y_train_orig = scalerY.inverse_transform(Y_train_scaled)
print('Pred train', mu_train)
print('Train', Y_train_orig)

# Save trained model and scaler
os.makedirs(dir_tostore, exist_ok=True)
pickle.dump(gpflow.utilities.parameter_dict(m), open(f'{dir_tostore}/gpflow_model_params{file_suffix}.p', "wb"))
pickle.dump(scalerX, open(f'{dir_tostore}/scalerX{file_suffix}.p', "wb"))
pickle.dump(scalerY, open(f'{dir_tostore}/scalerY{file_suffix}.p', "wb"))
pickle.dump(X_train_orig, open(f'{dir_tostore}/X_train{file_suffix}.p', "wb"))
pickle.dump(Y_train_orig, open(f'{dir_tostore}/Y_train{file_suffix}.p', "wb"))

print_summary(m)
