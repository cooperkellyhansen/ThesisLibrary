"""""
Author: Cooper Hansen
Data: Gary Whelan PhD

"""""

import SVE
from SVE import *
from Ensemble import *
from Orientation import Orientation
from HCPGrain import HCPGrain
from HCPTexture import HCPTexture
import os

import sklearn as skl
from sklearn import *
import sklearn.model_selection as ms
from sklearn.preprocessing import *
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import r2_score as r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


import pandas as pd
import numpy as np
from numpy import cosh,sinh,log,exp,sqrt
from numpy.linalg import *
import knn as knn
import matplotlib.pyplot as plt
from statistics import mean
import scipy
from scipy.stats import genextreme

#feature engineering stuff
#import feature_selection as fs
import run_bingo as bgo
#import featuretools as ft


#Normalization
def normalizeit(X):
    # scalar object
    scaler = StandardScaler()
    X = normalize(scaler.fit_transform(X))
    return X

########################################################################################################################
## BUILD ENSEMBLE
#instantiate Ensemble object
loadingB = object
path = 'IN625/Loading_Scenario_A/loadingA'
with open(path, 'rb') as f:
    loadingB = pickle.load(f)

#loadingA = Ensemble()
#loadingB = Ensemble()
#loadingA.fromSVEEnsemble(loading_scenario='A',structure_type='FCC',avg_scheme='grain')
#loadingB.fromSVEEnsemble(loading_scenario='B',structure_type='FCC',avg_scheme='grain')


# EV stats
#loadingA.generalizedEV(num_fips=1000,analysis_type='ensemble')
#loadingA.generalizedEV(num_fips=100,analysis_type='individual')
########################################################################################################################
# ANALYZE DATA
columns = ['Max Schmid of Grain','Misorientation','Shared Surface Area','Grain Size','mP Slip Trans',
                        'Sphericity','Delta Max Schmid','Delta Grain Size']
#columns = ['Max Schmid of Grain', 'Grain Size','Delta Max Schmid','Delta Grain Size']
X,y,micro_data = loadingB.analyze(desired_data=[0,1,2,3,4,5,6,7],
                  cols=columns,
                  structure_type='FCC',
                  hiplot=True,
                  bingo=True,
                  featuretools=True,
                  EV=False,
                  weight=False,
                  mean_homog=True) # current max not mean
X = normalizeit(X)
########################################################################################################################
## BINGO SUPER FEATURES
# only Schmid and Delta Schmid as inputs
#X = np.stack((X[:,0],X[:,1]),axis=-1)
#fs.main(X,y)

#Bingo run with feat features
#X = np.asarray(X)
#y = np.asarray(y)
#X_0 = X[:,0]
#X_1 = X[:,1]
#X_2 = X[:,2]
#X_3 = X[:,3]
#X_4 = X[:,4]
#X_5 = X[:,5]
#X_6 = X[:,6]
#X_7 = X[:,7]

#X_0s = log(0.6261*X_7)
#X_1s = (0.4668*X_6*X_1)
#X_2s = exp(0.4579*X_6)

#X = np.stack((X_0s,X_1s,X_2s),axis=-1)


#Normal bingo run
bgo.main(X,y)

#pred = -18.021111 + (6.830460)*(X_1s) + (100.830851 + (0.983479)*((-34.095161 + (-2)*(X_1s) + (7.310745)*(((X_2s)**(-1))*(log(sqrt(X_2s)))))*(log(7.310745) + sqrt(X_2s))))*(sqrt((7.310745)**(((X_2s)**(-1))*(log(sqrt(X_2s))))))
#loadingA.parityPlot(y,superfeatures=[pred],title='Super Features')



# ensemble1.SVE_reconstruct(1,1)


#####################################################################################################################
#
