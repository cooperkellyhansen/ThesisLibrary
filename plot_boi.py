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





#columns = ['Max Schmid of Grain','Misorientation','Shared Surface Area','Grain Size','mP Slip Trans',
#                                'Sphericity','Delta Max Schmid','Delta Grain Size']
columns = ['Max Schmid of Grain','Grain Size','Delta Max Schmid','Delta Max Schmid']
X,y,micro_data = loadingA.analyze(desired_data=[0,3,6,7],
                          cols=columns,
                          structure_type='FCC',
                          hiplot=True,
                          bingo=True,
                          featuretools=True,
                          EV=False,
                          weight=False,
                          mean_homog=True) # current max not mean
X = normalizeit(X)
y = normalizeit(y)
print(y)

# split 
X_0 = X[:,0]
X_1 = X[:,1]
X_2 = X[:,2]
X_3 = X[:,3]
#X_4 = X[:,4]
#X_5 = X[:,5]
#X_6 = X[:,6]
#X_7 = X[:,7]

#put equations here
pred = 6570002.802420 + (8.882155)(X_1 + (0.012791)(-57829227.501058 + (54.909165)((X_1)(X_1)) + log(X_1 - (log(X_2))) + (-242.360179)((log(X_2) + (0.581494)(((X_2)^(-1))(X_1 - (log(X_2)))))(sqrt(log(X_2))))))
loadingA.parityPlot(y=y,superfeatures=[pred],title='Test Plot DSchmid and Schmid')
