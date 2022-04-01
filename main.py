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
from numpy import cos,sin
from numpy.linalg import *
import knn as knn
import matplotlib.pyplot as plt
from statistics import mean
import scipy
from scipy.stats import genextreme

#bingo stuff
#import feature_selection as fs
#import run_bingo as bgo

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
X,y = ensemble1.analyze(desired_data=[0,1,2,3,4,5,6,7], #8 is the polytexture
                  cols=['Max Schmid of Grain','Misorientation','Shared Surface Area','Grain Size','mP Slip Trans',
                        'Sphericity','Delta Max Schmid','Delta Grain Size','Global Texture Parameter'],
                  structure_type='HCP',
                  sample_num=1,
                  hiplot=True,
                  bingo=True,
                  EV=True,
                  weight=False,
                  mean_homog=True) # current max not mean
########################################################################################################################
## BINGO SUPER FEATURES

# #plot iterations of super feature
# ensemble1.parityPlot(superfeatures=[X_2,X_3,X_4,X_5,X_6],title='Test Plot DSchmid and Schmid')

#Normal bingo run
# Normalization
def normalizeit(X):
    # scalar object
    scaler = StandardScaler()
    X = normalize(scaler.fit_transform(X))
    return X

X = normalizeit(X)
#bgo.main(X,y)

#Bingo run with superfeature
X = np.asarray(X)
y = np.asarray(y)
#fs.main(X,y)
# X_0 = X[:,0]
# X_1 = X[:,6]
# X_2 = (2.3040377806794283)*((0.4995016856895523)**(cos(X_0)) + ((X_1)**(-1))*(-1.7388482812229057e-05 + (0.007977149998994622)*((1 + (1534.8088852099588)*(np.add(X_0, X_1)))**(-1))))
# X_3 = X_2 + (1.9265498800081235e-05)*((sin(-3036.0083593656136 + X_2) - (((X_2)**(-1))*(sin(-3036.0083593656136 + X_2))))**(-1))
# X_4 = -5.924865604796324 + (-12.366480849119277)*((X_3)*(-1.5094102737631774 + (1.1898207272012027)*(X_3) + sin(sin(X_3)) - (X_3)))
# X_5 = (0.997636661566776 + ((5795.690703623)**(-1))*(X_2))*(X_4) + ((X_0)**(-1))*(((X_3)**(-1))*((cos(cos((5795.690703623)*(X_2)) - (X_0)))**(7478884.060552109)))
# X_6 = X_5 + ((-1993227.6303660949 + (489759.59863060905)*((X_0)**(-1)))**(-1))*(((X_0)**(-1))*(X_5))

X_0 = X[:,0]
X_1 = X[:,1]
X_2 = X[:,2]
X_3 = X[:,3]
X_4 = X[:,4]
X_5 = X[:,5]
X_6 = X[:,6]
X_7 = X[:,7]
X_8 = X[:,8]


pred = 1.179917 + (-0.078000)*((-2)*(X_3) + X_7 + (1 - (X_7))*((2)*(X_7) - (X_3)) + ((X_2)*((X_6)**(-1)))*((X_7)*((-0.102331 + (2)*(X_7) - (X_3))*(cos((1 - (X_7))*(-4961.663285 + X_7 + (-9864.451891)*(X_8) - (X_6)))))))
ensemble1.parityPlot(superfeatures=[pred],title='Test Plot DSchmid and Schmid')
#X_sup = np.stack((X_0,X_1,X_2,X_3,X_4,X_5,X_6),axis=-1)
X = np.delete(X,(0,6),axis=1)
X_6 = np.reshape(X_6,(len(X_6),1))
X = np.append(X,X_6,axis=1)
print(X.shape)
#bgo.main(X,y)



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
