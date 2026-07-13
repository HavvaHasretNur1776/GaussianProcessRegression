#!/bin/python

####################
import os, sys

year  = 2016#2016#2017#2018
magnets = ['up','down']
'''components = [
    "signal"#,
    
]
'''
#################### Import things
seed=120
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
from sklearn.model_selection import train_test_split
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches
sns.set(font_scale=0.9) 
sns.set(style="ticks", color_codes=True)
mpl.rcParams.update({'font.size': 12.5})
plt.rcParams['axes.unicode_minus'] = False

from util_resolution import return_phasespace, Plot, MSE, return_quadsolution, get4mom, return_boostapprox, get_fitvars
from GPFlow_Model import Model
import amplitf.kinematics as atfk

# build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
####################

training_vars = ['dst_PX', 'dst_PY', 'dst_PZ', 'mu_PX', 'mu_PY', 'mu_PZ', 'b0_Flight_X', 'b0_Flight_Y', 'b0_Flight_Z']

target_vars   = []
target_vars  += ['b0_P_TRUE']
####################


#Define masses
mB = 5279.65000000
mDst = 2010.27000000 
ml  = 105.65837121
mTau= 1776.81991209 

####################
component = "b0_dstmunu"
#for component in components:
for mag in magnets:
    print(f"\n=== Processing component: {component} Mag {mag} ===")
    dir_tostore = f'{component}/pickled_{year}_{mag}'
    os.makedirs(f"{dir_tostore}/plots", exist_ok=True)

    # read full dataset
    df = pd.read_pickle(f'./{component}/preppedData_{year}_{mag}.p')
    print("Loaded dataset:", df.shape)

    # get saved model and scaler from Tracker Only
    os.system(f'cp ./TO_trained_model/scalerX.p {dir_tostore}/')
    os.system(f'cp ./TO_trained_model/scalerY.p {dir_tostore}/')
    os.system(f'cp ./TO_trained_model/gpflow_model_params.p {dir_tostore}/')
    os.system(f'cp ./TO_trained_model/X_train.p {dir_tostore}/')
    os.system(f'cp ./TO_trained_model/Y_train.p {dir_tostore}/')
    
    '''os.system(f'cp ./FS_trained_model_v2/scalerX.p {dir_tostore}/')
    os.system(f'cp ./FS_trained_model_v2/scalerY.p {dir_tostore}/')
    os.system(f'cp ./FS_trained_model_v2/gpflow_model_params.p {dir_tostore}/')
    os.system(f'cp ./FS_trained_model_v2/X_train.p {dir_tostore}/')
    os.system(f'cp ./FS_trained_model_v2/Y_train.p {dir_tostore}/')'''
   
    scalerX = pickle.load(open(f'{dir_tostore}/scalerX.p', "rb"))
    scalerY = pickle.load(open(f'{dir_tostore}/scalerY.p', "rb"))
    params  = pickle.load(open(f'{dir_tostore}/gpflow_model_params.p', "rb"))

    # load the training data used to define the model (needed for correct shapes)
    X_train = pickle.load(open(f'{dir_tostore}/X_train.p', "rb"))
    Y_train = pickle.load(open(f'{dir_tostore}/Y_train.p', "rb"))

    # scale training data (for model initialization only)
    X_train = scalerX.transform(X_train)
    Y_train = scalerY.transform(Y_train)

    # build model and load saved parameters
    m = Model(X_train, Y_train, seed, params)

    # ---- apply to the FULL dataset ----
    X_full = df[training_vars].values
    Y_full = df[target_vars].values  # only if you want truth for comparison
    X_full = scalerX.transform(X_full)

    # predict on full dataset
    mu_full, var_full = m.predict_y(X_full)
    mu_full = scalerY.inverse_transform(mu_full)

    # attach predictions to dataframe
    df['b0_P_Estm'] = mu_full[:, 0]

    # get Mu 4-momentum
    '''Pl_lab = atfk.lorentz_vector(
        atfk.vector(df['Lepton_PX_Reco'], df['Lepton_PY_Reco'], df['Lepton_PZ_Reco']),
        df['Lepton_E_Reco']
    )'''#commented out b/c already added in util_resolution.py/get_fitvars()
    p3_Bfd = atfk.vector(df['b0_Flight_X'], df['b0_Flight_Y'], df['b0_Flight_Z'])

    # compute fit variables
#    df = get_fitvars(df, Pl_lab, p3_Bfd, mB, '')
    df = get_fitvars(df, p3_Bfd, mB)
    df = df.dropna()

    # compute w
    df['w_Pred'] = (mB**2 + mDst**2 - df['q2_Pred']) / (2*mB*mDst)

    # save regressed dataset
    outname = f'{dir_tostore}/regressed.p'
    df.to_pickle(outname)
    print("Saved:", outname)

    # make plots
    Plot(df=df, var='b0_P', suffix='', Dir=f'{dir_tostore}/plots', log=False, res_limits_rel=(-1.,1.))
    Plot(df=df, var='q2',   suffix='', Dir=f'{dir_tostore}/plots', log=False, res_limits_rel=(-1.,1.))
    Plot(df=df, var='w',    suffix='', Dir=f'{dir_tostore}/plots', log=False, res_limits_rel=(-1.,1.))
    Plot(df=df, var='chi',    suffix='', Dir=f'{dir_tostore}/plots', log=False, res_limits_rel=(-1.,1.))
    Plot(df=df, var='costhetal',    suffix='', Dir=f'{dir_tostore}/plots', log=False, res_limits_rel=(-1.,1.))
    Plot(df=df, var='costhetad',    suffix='', Dir=f'{dir_tostore}/plots', log=False, res_limits_rel=(-1.,1.))

    
    Plot(df=df, var='b0_P', suffix='', Dir=f'{dir_tostore}/plots', log=True, res_limits_rel=(-1.,1.))
    Plot(df=df, var='q2',   suffix='', Dir=f'{dir_tostore}/plots', log=True, res_limits_rel=(-1.,1.))
    Plot(df=df, var='w',    suffix='', Dir=f'{dir_tostore}/plots', log=True, res_limits_rel=(-1.,1.))
    Plot(df=df, var='chi',    suffix='', Dir=f'{dir_tostore}/plots', log=True, res_limits_rel=(-1.,1.))
    Plot(df=df, var='costhetal',    suffix='', Dir=f'{dir_tostore}/plots', log=True, res_limits_rel=(-1.,1.))
    Plot(df=df, var='costhetad',    suffix='', Dir=f'{dir_tostore}/plots', log=True, res_limits_rel=(-1.,1.))

    print(f"Plots saved in {dir_tostore}/plots")
