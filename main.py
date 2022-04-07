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
#import run_bingo as bgo
import featuretools as ft


#Normalization
@staticmethod
def normalizeit(X):
    # scalar object
    scaler = StandardScaler()
    X = normalize(scaler.fit_transform(X))
    return X

########################################################################################################################
## BUILD ENSEMBLE
#instantiate Ensemble object
loadingA = object
path = 'IN625\Loading_Scenario_A\loadingA'
with open(path, 'rb') as f:
    loadingA = pickle.load(f)

# loadingA = Ensemble()
# loadingB = Ensemble()
# loadingA.fromSVEEnsemble(loading_scenario='A',structure_type='FCC')
# loadingB.fromSVEEnsemble(loading_scenario='B',structure_type='FCC')

# EV stats
#loadingA.generalizedEV(num_fips=1000,analysis_type='ensemble')
#loadingA.generalizedEV(num_fips=100,analysis_type='individual')
########################################################################################################################
# ANALYZE DATA
columns = ['Max Schmid of Grain','Misorientation','Shared Surface Area','Grain Size','mP Slip Trans',
                        'Sphericity','Delta Max Schmid','Delta Grain Size']
#columns = ['Max Schmid of Grain','Delta Max Schmid']
X,y,micro_data = loadingA.analyze(desired_data=[0,1,2,3,4,5,6,7],
                  cols=columns,
                  structure_type='FCC',
                  hiplot=True,
                  bingo=True,
                  featuretools=True,
                  EV=False,
                  weight=False,
                  mean_homog=True) # current max not mean
########################################################################################################################
# ## FEATURETOOLS SUPERFEATURES
# FIP_data = pd.DataFrame(y,columns=['FIPs'])
#
# micro_data.reset_index(inplace=True)
# FIP_data.reset_index(inplace=True)
# micro_data = micro_data.rename(columns = {'index':'grain_num'})
#
# micro_data.reset_index(inplace=True)
#
# #initialize entity set
# es = ft.EntitySet(id="microstructure_data")
# #add both microstructure and FIP dfs
# es = es.add_dataframe(
#     dataframe_name="micro_data",
#     dataframe=micro_data,
#     index="index",
# )
# es = es.add_dataframe(
#     dataframe_name="FIP_data",
#     dataframe=FIP_data,
#     index="index",
# )
# # add relationship
# es = es.add_relationship('FIP_data','index','micro_data','grain_num')
#
# feature_matrix, feature_defs = ft.dfs(entityset=es,
#                                       target_dataframe_name="FIP_data",
#                                       agg_primitives=["mean", "sum", "mode"],
#                                       trans_primitives=["month", "hour"],
#                                       max_depth=3)
#
# print(feature_matrix)

#split data
# X_train, X_test, y_train, y_test = split(X,y,test_size=0.1)
# X_train_n, X_test_n = normalizeit(X_train,X_test)
#
# grain_pairs_y = np.array(grain_pairs_y).reshape((sve0.num_grains,1))
# scaler = MinMaxScaler(feature_range=(1,10))
# grain_pairs_y = scaler.fit_transform(grain_pairs_y)




########################################################################################################################
## BINGO SUPER FEATURES

# #plot iterations of super feature
# ensemble1.parityPlot(superfeatures=[X_2,X_3,X_4,X_5,X_6],title='Test Plot DSchmid and Schmid')

#Normal bingo run

#

# #bgo.main(X,y)

#Bingo run with superfeature
# X = np.asarray(X)
# y = np.asarray(y)
# X_0 = X[:,0]
# X_1 = X[:,6]
# X_2 = (-4.7383766443311626e-05) * (-22500.12339264026 + (((X_0 + X_1) ** (-1)) * (sinh(X_0)) - (51.448757103062206)) * (
#             51.448757103062206 + X_1 - (((X_0 + X_1) ** (-1)) * (sinh(X_0)))))
# X_3 = X_0 + (0.8421706779616214) * (0.9554151727208404 + (0.0013535941064905192) * (X_0 + exp(X_2)) - (X_1))
# X_4 = (0.9924117165718074) * (X_3 + (-0.000161136778045409) * (((X_0) ** (-1)) * ((sqrt(X_3) - (X_2)) ** (-1))))
# #define superfeat
# X_sup = X_4
# X_0 = X[:,0]
# X_1 = X[:,1]
# X_2 = X[:,2]
# X_3 = X[:,3]
# X_4 = X[:,4]
# X_5 = X[:,5]
# X_6 = X[:,6]
# X_7 = X[:,7]
# X = np.stack((X_1,X_2,X_3,X_4,X_5,X_7,X_sup),axis=-1)
# #X = normalizeit(X)
# X_0 = X[:,0]
# X_1 = X[:,1]
# X_2 = X[:,2]
# X_3 = X[:,3]
# X_4 = X[:,4]
# X_5 = X[:,5]
# X_6 = X[:,6]
#
#
# pred = -16.827412 + (-0.966902)*(((X_5 - (X_2))*(sqrt((X_5 - (X_2))*(cosh(X_0)))))*((cosh(X_0))*(-1.265151 + X_6 + ((-0.000322)*((X_6)**(-1)) + exp(X_2))*((0.630696 + (-0.067429)*(log(X_2)))*(cosh(X_2))) + (1.208249 + (X_3)*(sqrt(exp(X_2)) + (X_2 + X_6)*((sqrt((X_2 + X_3)*(cosh(X_3))))**(-1))))*(cosh(X_5)))))
# ensemble1.parityPlot(superfeatures=[pred],title='Super Features')
#
# ensemble1.SVE_reconstruct(1,1)


########################################################################################################################


#
# #split data
# X_train, X_test, y_train, y_test = split(X,y,test_size=0.1)
# X_train_n, X_test_n = normalizeit(X_train,X_test)
#
# grain_pairs_y = np.array(grain_pairs_y).reshape((sve0.num_grains,1))
# scaler = MinMaxScaler(feature_range=(1,10))
# grain_pairs_y = scaler.fit_transform(grain_pairs_y)
########################################################################################################################
