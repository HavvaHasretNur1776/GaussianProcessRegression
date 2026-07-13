import gpflow
import pandas as pd
import numpy as np
from gpflow.utilities import print_summary

def Model(x_train, y_train, seed, params = None):
    ############
    N = y_train.shape[0]  # number of points
    D = x_train.shape[1]  # number of input dimensions
    P = y_train.shape[1]  # number of observations = output dimensions
    ############

    ###################
    #build model
    kernel = gpflow.kernels.Matern52(variance=1., lengthscales = np.ones(D)) + gpflow.kernels.Linear() + gpflow.kernels.Bias()
    meanf  = gpflow.mean_functions.Linear(np.random.RandomState(int(seed)).randn(D, P))
    ###################

    ###################
    ##GPR
    #m = gpflow.models.GPR(X_train, y_train, kern=kernel, mean_function=meanf)
    #m.compile()
    ###################

    ###################
    #SGPR
    M = 4000  # number of inducing points
    inducing_pts = x_train[:M, :].copy()
    inducing_variable = gpflow.inducing_variables.InducingPoints(inducing_pts)
    m = gpflow.models.SGPR(data=(x_train, y_train), kernel=kernel, mean_function=meanf, inducing_variable=inducing_variable)
    print_summary(m)
    ###################

    if params:
        print('Loading the params')
        gpflow.utilities.multiple_assign(m, params)
        print_summary(m)
    
    return m
####################

