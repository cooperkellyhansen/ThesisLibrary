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
from numpy.linalg import *
import knn as knn
import matplotlib.pyplot as plt
from statistics import mean
import scipy
from scipy.stats import genextreme
#import feature_selection

########################################################################################################################
## BUILD ENSEMBLE

structure_type = 'HCP'
#instantiate Ensemble object
ensemble1 = Ensemble()
#ensemble1.fromSVEEnsemble(sample_num=1,structure_type='HCP')

# EV stats
#ensemble1.generalizedEV(fname='sample_1/data/sub_band_averaged_max_per_grain.csv', num_fips=1000,analysis_type='ensemble')
#ensemble1.generalizedEV(fname='sample_1/data/sub_band_averaged_max_per_grain.csv', num_fips=100,analysis_type='individual')
########################################################################################################################
## ANALYZE DATA
ensemble1.analyze(desired_data=[0,4,6,7],
                  cols=['Max Schmid of Grain','mP Slip Trans','Delta Max Schmid','Delta Grain Size','Global Texture Parameter'],
                  structure_type='HCP', sample_num=1, hiplot=True, EV=True, weight=True,weighting_feature_idx=2)


########################################################################################################################
# # Normalization
# def normalizeit(X_train, X_test):
#     # scalar object
#     scaler = StandardScaler()
#     X_train_n = normalize(scaler.fit_transform(X_train))
#     X_test_n = normalize(scaler.fit_transform(X_test))
#
#     return X_train_n, X_test_n
#
# #split data
# X_train, X_test, y_train, y_test = split(X,y,test_size=0.1)
# X_train_n, X_test_n = normalizeit(X_train,X_test)
#
# grain_pairs_y = np.array(grain_pairs_y).reshape((sve0.num_grains,1))
# scaler = MinMaxScaler(feature_range=(1,10))
# grain_pairs_y = scaler.fit_transform(grain_pairs_y)
########################################################################################################################
