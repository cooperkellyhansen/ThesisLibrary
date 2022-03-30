import numpy as np

from SVE import *

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import genextreme
from tqdm import tqdm
import itertools
import sklearn as skl
from sklearn.preprocessing import *




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
    def fromSVEEnsemble(self, sample_num,structure_type='FCC'):
        '''
        Builds each SVE in an ensemble and stores it in a pickle file.
        fname = file name string.
        structure_type = crystal structure type (FCC, HCP supported).
        '''

        # get files in folder
        path = 'sample_{}\data'.format(sample_num)  # sample name
        csv_files_feature = glob.glob(os.path.join(path, "*.csv")) # sve names

        # loop over the list of csv files and set features for each SVE
        skip_rows = 0
        for idx,f in enumerate(tqdm(csv_files_feature, desc ="Building SVE ensemble and dumping to pickle")):
            if f.startswith('sample_{}\data\FeatureData'.format(sample_num)):
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
                self.sveDict[key].set_sub_band_data('sample_{}\data\sub_band_averaged_max_per_grain.csv'.format(sample_num),skip_rows)
                skip_rows += self.sveDict[key].num_grains

                # create pkl for each sve and store in folder
                with open('sample_{}\SVE_Pickles\sve_{}.pkl'.format(sample_num,idx), 'wb') as f:
                    pickle.dump(self.sveDict[key],f)


        return None

########################################################################################################################
    def generalizedEV(self,fname,num_fips=100,analysis_type='ensemble'):
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

            #grab fips from file
            df = pd.read_csv(fname,header=None, names=['FIP', 'Grain', 'Slip', 'LayerBand', 'SubBand', 'SVE', 'Iteration'])
            df.set_index('Grain',inplace=True)
            fips = df['FIP'].tolist()
            top_fips = sorted(fips, reverse=True)[:num_fips]
            self.minFIP = min(top_fips)
            print('len top fips:', len(top_fips))
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
            # TODO: for some reason the higher the FIP the higher the probability (ask gary about this)
            fips = []
            plt.figure()
            plt.tight_layout()

            #pick some random SVEs
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
                print('FIPs',top_fips)
                print('max FIP', max(top_fips))
                print('max ex', max_extreme)

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
    def FilterByEVFIPs(self, sample_num=1):

        # Open saved SVE objects
        EV_grainNames = []
        path = 'sample_{}/SVE_Pickles'.format(sample_num)
        for idx, pkl_file in enumerate(os.listdir(path)):
            with open(os.path.join(path, pkl_file), 'rb') as f:
                sve_obj = pickle.load(f)
                # find grain names of EV FIPs
                # TODO: change the threshold to the one found in SVE class. Must re-run
                grainNames_cur = [(num,k) for num, (k, v) in enumerate(sve_obj.max_fips.items(),start=1) if v >= 1.5E-8]
                EV_grainNames.append(grainNames_cur)

        return EV_grainNames

########################################################################################################################
    def analyze(self,desired_data=[],cols=[], structure_type='FCC', sample_num=1, weight=False, weighting_feature_idx=0,
                bingo=False, hiplot=False, EV=True):

        # build ensemble from pkl files
        path = 'sample_{}/SVE_Pickles'.format(sample_num)
        for idx, pkl_file in enumerate(os.listdir(path)):
            with open(os.path.join(path, pkl_file), 'rb') as f:
                sve_obj = pickle.load(f)
                self.addSVE(sve_obj, idx)

        # set grain names for either grains with EV FIPs or all
        if EV:
            grainNames = self.FilterByEVFIPs(sample_num)
        elif not EV:
            # TODO: This needs to be a 2D array
            grainNames = [k for k, v in sve_obj.max_fips.items()]

        # loop thru SVEs in ensemble
        X = []
        y = []
        for idx, (sve_num, sve_obj) in enumerate(tqdm(self.sveDict.items(), desc='Filtering for EV and building data set')):

            # grab texture object
            if structure_type == 'FCC':
                texture = sve_obj.sveFCCTexture
            elif structure_type == 'HCP':
                texture = sve_obj.sveHCPTexture

            # volume-weighted average of the Schmid factor for the entire poly-crystal
            poly_texture = sum([texture.primary_slip[grain][0] * volume for grain,volume in
                                sve_obj.volume.items()]) / sum(list(sve_obj.volume.values()))

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
                schmid = [texture.primary_slip['Grain_{}'.format(neighbor)][0]  for neighbor in neighbors]
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
                    # mean of delta features
                    features[6] = mean(features[6])
                    features[7] = mean(features[7])

                # gather desired features and valid FIPs
                X_cur = []
                for idx in desired_data:
                    if idx in desired_data:
                        X_cur.append(features[idx])
                X_cur.append(poly_texture)
                X.append(X_cur)
                y.append([sve_obj.max_fips[grain_name]])


        if bingo:
            # scale FIPs
            y = np.array(y).reshape((len(y), 1))
            scaler = MinMaxScaler(feature_range=(1, 10))
            y = scaler.fit_transform(y)

            return X, y

        if hiplot:
            # create and return csv
            df = pd.DataFrame(X, columns=cols)
            df['FIP'] = [item for sublist in y for item in sublist]
            df.to_csv('hi.csv', index=False)

