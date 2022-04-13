import numpy as np

from SVE import *

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import genextreme
import itertools

import slice_sve as ss


class Ensemble:
    '''
    This class is a representaion of a sample group of SVE's (Statistical Volume Element). It is a wrapper class for
    the SVE class which only represents one SVE The object is built from csv files that are output by DREAM.3D software.
    There are methods to clean data, grab data, and manipulate data. It also uses the Orientation and FCCGrain classes
    and FCCTexture wrapper class written by Dr. Jacob Hochhalter.
    '''

    def __init__(self):
        self.sveDict = {}
        self.minFIP = float
        self.EV_X = []
        self.EV_y = []

########################################################################################################################
    def addSVE(self, sve_obj, sve_num):
        '''
        add an HCPGrain object to the HCPTexture object
        hcpo_grain = HCPGrain object
        grain_name = any hashable type to be used as dictionary key
        '''
        SVE_name = 'sve_{}'.format(sve_num)
        self.sveDict[SVE_name] = sve_obj
    # TODO: add a to_file boolean that gives the option to put into a file instead of default.

########################################################################################################################
    def fromSVEEnsemble(self,structure_type='FCC',loading_scenario='A',avg_scheme='element'):
        '''
        Builds each SVE in an ensemble and stores it in a pickle file.
        fname = file name string.
        structure_type = crystal structure type (FCC, HCP supported).
        '''

        # get files in folders
        csv_files_feature = []
        for i in range(1,41):
            # TODO: path should probably be an input
            path = 'IN625/Loading_Scenario_{}/{}{}'.format(loading_scenario,loading_scenario,i)  # loading scenario
            csv_files_feature.append(glob.glob(os.path.join(path, "*.csv"))) # sve names

        # loop over the list of csv files and set features for each SVE
        for idx,files in enumerate(csv_files_feature,start=1):
            for f in files:
                if f.startswith('IN625/Loading_Scenario_{}/{}{}/FeatureData'.format(loading_scenario,loading_scenario,idx)):
                    key = 'sve_{}'.format(idx)
                    self.sveDict[key] = SVE()
                    # set features from feature data file
                    self.sveDict[key].textureFromEulerAngles(f,structure=structure_type)
                    self.sveDict[key].set_features(f)
                    self.sveDict[key].set_grain_neighbors(f)
                    self.sveDict[key].set_surface_area(f)
                    self.sveDict[key].calc_schmidFactors(structure_type,file_type='euler')
                    self.sveDict[key].calc_misorientations(structure_type)
                    self.sveDict[key].calc_mPrime()
                    self.sveDict[key].set_sub_band_data('IN625/Loading_Scenario_{}/{}{}/max_grain_FIPs.csv'.format(loading_scenario,loading_scenario,idx))
                    if avg_scheme == 'grain':
                        self.sveDict[key].set_grain_element_data('IN625/Loading_Scenario_{}/{}{}/FIP_df.csv'.format(loading_scenario,loading_scenario,idx))
                    #if structure_type == 'FCC':
                        # add in Bishop-Hill calculated Taylor Factor
                        #if f.startswith('IN625\Loading_Scenario_{}\{}{}\FIP'.format(loading_scenario,loading_scenario,idx)):
                            #self.sveDict[key].calc_taylorFactor()

                    # # create pkl for each sve and store in folder
                    # with open('IN625\Loading_Scenario_{}\SVE_Pickles\sve_{}.pkl'.format(sample_num,idx), 'wb') as f:
                    #     pickle.dump(self.sveDict[key],f)

        # create pkl for ensemble
        with open('IN625/Loading_Scenario_{}/loading{}'.format(loading_scenario,loading_scenario), 'wb') as f:
            pickle.dump(self, f)

        return None

########################################################################################################################
    def generalizedEV(self,num_fips=100,analysis_type='ensemble'):
        '''
        calculates the generalized extreme value distribution for the
        sub band averaged max FIPs. These distributions can be calculated
        for the entire ensemble ('ensemble') or for each SVE ('individual').
        The number of FIPs denotes the number of highest fips to use in the
        fit. Theoretically if the number of top FIPs is too large, the data
        will not fit the Frechet distribution very well.
        :return: None. This however will plot the desired fit.
        '''

        if analysis_type == 'ensemble':
            plt.figure()
            plt.tight_layout()

            ensemble_FIPs = []
            for sve_obj in self.sveDict.values():
                ensemble_FIPs.extend(sve_obj.max_fips.values())

            top_fips = sorted(ensemble_FIPs, reverse=True)[:num_fips]
            self.minFIP = min(top_fips)
            # calculate GEV fit
            fit = genextreme.fit(top_fips)

            # GEV parameters from fit
            c, loc, scale = fit
            #fit_mean = loc
            min_extreme, max_extreme = genextreme.interval(0.999, c, loc, scale)

            # evenly spread x axis values for pdf plot
            x = np.linspace(min_extreme, max_extreme, num_fips)

            # plot distribution
            y = -1 * (genextreme.logcdf(x, *fit))
            plt.scatter(np.log(x), -np.log(y), marker='^')
            # plt.xscale('log')
            plt.xlabel('ln(FIP)')
            plt.ylabel('-ln(-ln(p))')
            plt.title('Frechet Distribution for Max {} FIPs in Ensemble'.format(num_fips))
            # plt.hist(top_fips, 50, alpha=0.3)
            plt.show()


        elif analysis_type == 'individual':
            fips = []
            plt.figure()
            plt.tight_layout()

            randomSVElist = [random.choice(list(self.sveDict)) for i in range(0,5)]
            marker = itertools.cycle((',', '+', '.', 'o', '*'))

            for idx,SVE in enumerate(tqdm(randomSVElist, desc='Fitting sampled SVEs to GEV')): # self.sveDict.values()

                top_fips = sorted(self.sveDict[SVE].max_fips.values(), reverse=True)[:num_fips]
                # calculate GEV fit
                top_fips.reverse()
                fit = genextreme.fit(top_fips)

                # GEV parameters from fit
                c, loc, scale = fit
                #fit_mean = loc
                min_extreme, max_extreme = genextreme.interval(0.999, c, loc, scale)

                # evenly spread x axis values for pdf plot
                x = np.linspace(min_extreme, max_extreme, num_fips)

                # plot distribution
                y = -1*(genextreme.logcdf(x,*fit))
                plt.scatter(np.log(x), -np.log(y), marker=next(marker),alpha=0.5,label='SVE_{}'.format(idx))
                #plt.xscale('log')
                plt.xlabel('ln(FIP)')
                plt.ylabel('-ln(-ln(p))')
                plt.title('Frechet Distribution for Max {} FIPs in 10 sampled SVEs'.format(num_fips))
                plt.legend()
                #plt.hist(top_fips, 50, alpha=0.3)
            plt.show()

        return None

########################################################################################################################
    def FilterByEVFIPs(self, loading_scenario):

        # Open saved SVE objects
        EV_grainNames = []
        path = 'IN625\Loading_Scenario_{}\loading{}'.format(loading_scenario,loading_scenario)
        # find grain names of EV FIPs
        for (sve_num,sve_obj) in self.sveDict.items():
            # TODO: change the threshold to the one found in SVE class. Must re-run
            grainNames_cur = [(num,k) for num, (k, v) in enumerate(sve_obj.max_fips.items(),start=1) if v >= 1.5E-10]
            EV_grainNames.append(grainNames_cur)

        return EV_grainNames

########################################################################################################################
    def analyze(self, desired_data=[],cols=[], structure_type='FCC', loading_scenario='A', weight=False, weighting_feature_idx=0, mean_homog=False,
                bingo=False, hiplot=False,featuretools=False, EV=False):

        # set grain names for either grains with EV FIPs or all
        if EV:
            grainNames = self.FilterByEVFIPs(loading_scenario)
        elif not EV:
            grainNames = []
            # find grain names of EV FIPs
            for (sve_num, sve_obj) in self.sveDict.items():
                grainNames_cur = [(num, k) for num, (k, v) in enumerate(sve_obj.max_fips.items(), start=1)]
                grainNames.append(grainNames_cur)

        # loop thru SVEs in ensemble
        X = []
        y = []

        for idx, (sve_num, sve_obj) in enumerate(self.sveDict.items()):

            # grab texture object
            if structure_type == 'FCC':
                texture = sve_obj.sveFCCTexture
            elif structure_type == 'HCP':
                texture = sve_obj.sveHCPTexture

            #loop through grains in SVE (results in features of grain and its neighbors)
            for grain_num, grain_name in grainNames[idx]:
                # find valid neighbors
                grainNumbers = [num[0] for num in grainNames[idx]]
                neighbors = [neighbor for neighbor in sve_obj.grain_neighbor_link[grain_name]
                             if neighbor in grainNumbers]  # threshold neighbors schmid factor
                if not neighbors:
                    continue #if no valid neighbors exist, skip.

                # gather/calculate all currently known features
                # schmid factor of grain
                schmid = [texture.primary_slip['Grain_{}'.format(neighbor)][0] for neighbor in neighbors]
                # misorientations
                mis = [sve_obj.neighbor_misorient[grain_name][neighbor] for neighbor in neighbors]
                # shared surface area
                ssa = [sve_obj.neighbor_shared_surface_area[grain_name][neighbor] for neighbor in neighbors]
                # grain size
                gsize = [sve_obj.volume['Grain_{}'.format(neighbor)] for neighbor in neighbors]
                # m' slip transmission parameter
                mp = [sve_obj.neighbor_mp[grain_name][neighbor] for neighbor in neighbors]
                # sphericity of grain
                sphere = [sve_obj.omega3s['Grain_{}'.format(neighbor)] for neighbor in neighbors]
                # difference in Schmid between neighboring grains (neighbor - grain)
                dschmid = [(nschmid - texture.primary_slip[grain_name][0]) for nschmid in schmid]
                # difference in grain size between neighboring grains (neighbor - grain)
                dgsize = [(nvolume - sve_obj.volume[grain_name]) for nvolume in [sve_obj.volume['Grain_{}'.format(neighbor)] for neighbor in neighbors]]

                features = [schmid,mis,ssa,gsize,mp,sphere,dschmid,dgsize]

                if weight:
                    # weight the data
                    weighting_feature = features[weighting_feature_idx]
                    for i in range(0,len(features)-2):
                        features[i] = sum([features[i][idx] * weight for idx, weight in enumerate(weighting_feature)])\
                                      / sum(weighting_feature)
                        # features[i] = sum([features[i][idx] * poly_texture for idx in range(len(features[i]))])\
                        #                 / sum(poly_texture) # weighting schmid by poly_texture
                    # mean of delta features
                    features[6] = mean(features[6]) # delta schmid
                    features[7] = mean(features[7]) # delta grain size

                if mean_homog:
                    features = [mean(feature) for feature in features]
                    # features[0] # Schmid of grain
                    # features[1] # Misorientation
                    # features[2] # Shared surface area
                    # features[3] # Grain volume
                    # features[4] # M' parameter
                    # features[5] # Sphericity
                    # features[6] # delta Schmid
                    # features[7] # delta grain size

                # gather desired features and valid FIPs
                X_cur = []
                for idx in desired_data:
                    if idx in desired_data:
                        X_cur.append(features[idx])
                X.append(X_cur)
                # y.append([sve_obj.max_fips[grain_name]])
                # grain avg FIPsi
                y.append([sum(sve_obj.elem_grain_link['FIPs'][grain_num])/len(sve_obj.elem_grain_link['FIPs'][grain_num])])
                #print('sum/len:',[sum(sve_obj.elem_grain_link['FIPs'][grain_num])/len(sve_obj.elem_grain_link['FIPs'][grain_num])])
                #print('mean:',mean(sve_obj.elem_grain_link['FIPs'][grain_num]))
        #print(self.sveDict['sve_1'].elem_grain_link['element'][1])
        #print(self.sveDict['sve_1'].elem_grain_link['FIPs'][1])
        #print(sum(self.sveDict['sve_1'].elem_grain_link['FIPs'][1])/len(self.sveDict['sve_1'].elem_grain_link['FIPs'][1]))
        if featuretools:
            df = pd.DataFrame(X,columns=cols)

        if bingo:
            # scale FIPs
            y = np.log(np.asarray(y))
            #y = np.array(y).reshape((len(y), 1))
            #scaler = MinMaxScaler(feature_range=(1, 10))
            #y = scaler.fit_transform(y)
            #if EV:
            self.EV_X = np.asarray(X)
            self.EV_y = np.asarray(y)

        if hiplot:
            # create and return csv
            df = pd.DataFrame(X, columns=cols)
            df['FIP'] = [item for sublist in y for item in sublist]
            df.to_csv('hi_abs.csv', index=False)

        return X,y,df
########################################################################################################################
    def parityPlot(self,y,superfeatures=[],title=''):
        '''
        This function takes in the different superfeature iterations or a single prediction array from bingo and
        creates a parity plot between the actual max FIPs and the predicted max FIPs

        :param superfeatures: list of superfeature orientations
        :param title: Title of plot
        :return: None
        '''


        #plot actual FIPs
        plt.figure()
        plt.tight_layout()
        plt.scatter(y, y, marker='.',label='actual')
        plt.title(title)
        plt.xlabel('Actual Max EV FIPs')
        plt.ylabel('Predicted Max EV FIPs')

        #plot superfeatures
        marker = itertools.cycle(('^', '+', '.', 'o', '*'))
        for iter,feat in enumerate(superfeatures,start=1):
            plt.scatter(y,feat,marker=next(marker),alpha=0.5,label='iteration_{}'.format(iter))
        plt.legend()
        plt.savefig('/uufs/chpc.utah.edu/common/home/u0736958/Thesis/ThesisLibrary/parity_grain')
        plt.show()

        #plot fitness values

########################################################################################################################
    def SVE_reconstruct(self,sample_num=1,sve_num=0):
        '''
        Simply reconstruct an SVE using a 3d projection from matplotlib
        :param sample_num: number of the ensemble of SVEs (int)
        :param sve_num: number of the sve in the ensemble (int)
        :return:None
        '''


        path = 'sample_{}/SVE_Pickles/sve_{}.pkl'.format(sample_num,sve_num)
        with open(path, 'rb') as f:
            sve_obj = pickle.load(f)
            self.addSVE(sve_obj, sve_num)
        #grab grain centroids from SVE (position in plot)
        centroids = self.sveDict['sve_{}'.format(sve_num)].centroid.values()
        centroids = np.array(list(centroids))
        #grab grain volume from SVE (marker size)
        grain_volume = self.sveDict['sve_{}'.format(sve_num)].volume.values()
        grain_volume = np.array(list(grain_volume))
        #grab max grain FIP from SVE (color)
        FIPs = self.sveDict['sve_{}'.format(sve_num)].max_fips.values()
        FIPs = np.array(list(FIPs))
        #grab misorientaions (color)
        #misorientations =

        # plot
        fig = plt.figure()
        plt.tight_layout()
        ax = fig.add_subplot(111,projection='3d')

        #3d markers
        # #normalize volume
        # scaler = MinMaxScaler(feature_range=(0.1,2))
        # grain_volume = scaler.fit_transform(np.array(grain_volume).reshape((len(grain_volume),1)))
        # for (xi, yi, zi, ri, ci) in zip(centroids[:,0],centroids[:,1],centroids[:,2], grain_volume,FIPs):
        #     (xs, ys, zs) = self._drawSphere(xi, yi, zi, ri)
        #     ax.plot(xs, ys, zs, color='r')

        #2d markers
        p = ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c=FIPs,s=grain_volume,marker='.',alpha=1.0,cmap='jet',norm=colors.LogNorm())
        # set specimen coords
        ax.set_xlabel('X specimen')
        ax.set_ylabel('Y specimen')
        ax.set_zlabel('Z specimen')
        ax.set_title('SVE_{}'.format(sve_num))
        cbar = fig.colorbar(p)
        cbar.set_label('Max FIP')
        plt.show()


        # #scroll through 2d slices of plot
        # fig,ax = plt.subplots(1,1)
        # x,y = np.meshgrid(centroids[:,0],centroids[:,1])
        # #z = FIPs[x,y]
        # X = np.random.rand(20, 20, 40)
        #
        # scroll = ss.IndexTracker(ax,coords)
        # fig.canvas.mpl_connect('scroll_event', scroll.on_scroll)
        # plt.show()


    @staticmethod
    def _drawSphere(xCenter, yCenter, zCenter, r):
        #draw sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x=np.cos(u)*np.sin(v)
        y=np.sin(u)*np.sin(v)
        z=np.cos(v)

        # shift and scale sphere
        x = r*x + xCenter
        y = r*y + yCenter
        z = r*z + zCenter
        return (x,y,z)
