from Ensemble import *
from numpy import *
from sklearn.preprocessing import *

#Normalization
def normalizeit(X):
    # scalar object
    scaler = StandardScaler()
    X = normalize(scaler.fit_transform(X))
    return X



#load Ensemble
loadingA = object
path = 'IN625/Loading_Scenario_A/loadingA'
with open(path, 'rb') as f:
        loadingA = pickle.load(f)





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
X = normalizeit(X)



# split 
X_0 = X[:,0]
X_1 = X[:,1]
X_2 = X[:,2]
X_3 = X[:,3]
X_4 = X[:,4]
X_5 = X[:,5]
X_6 = X[:,6]
X_7 = X[:,7]

#put equations here


pred = -17.966084 + X_0 + (X_6)*((-2.717788 - (X_6))*(cosh(X_6)) - (X_3)) - (X_7)
# plot
loadingA.parityPlot(superfeatures=[pred],title='Test Plot DSchmid and Schmid')
