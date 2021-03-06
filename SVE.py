import HCPTexture,FCCTexture

from numpy import *
import pandas as pd
import pickle
from HCPTexture import *
from FCCTexture import *

pd.options.display.max_colwidth = 1500 # Some strings are long so this will fix truncation

class SVE:

    '''
    This class is a representaion of an SVE (Statistical Volume Element). The object is built from
    csv files that are output by DREAM.3D software. There are methods to clean data, grab data, and manipulate data.
    it also uses the Orientation and FCCGrain classes and FCCTexture wrapper class written by Dr. Jacob Hochhalter
    '''


    #############################################
    def __init__(self):
        '''
        Many of these SVE attributes are dictionaries. The key of the dictionary corresponds to the grain number (Feature_ID).
        The attributes are based off the output options of the DREAM.3D software and can be added to for various applications.
        '''

        self.num_grains = int
        self.num_elems = 27000 # standard for DREAM.3D SVEs
        self.sve_num = int

        self.elem_grain_link = {}
        self.grain_elem_stress = {} # all stress tensors for elements in grain
        self.grain_elem_strain = {}
        self.grain_boundary_nodes = {}
        self.node_coords = {}
        self.elem_coords = {}
        self.vAvg_stress = {}
        self.vAvg_strain = {}
        self.strain_range = {}

        self.shared_surface_area = {}
        self.neighbor_misorient = {}
        self.neighbor_mp = {}
        self.neighbor_shared_surface_area = {}

        self.quaternions = {}
        self.axis_euler = {}
        self.euler = {}
        self.centroid = {}
        self.volume = {}
        self.omega3s = {}
        self.axislengths = {}
        self.max_fips = {}  # max fip per grain
        self.taylorFactor = {}

        self.grain_neighbor_link = {}

        self.sveHCPTexture = HCPTexture()
        self.sveFCCTexture = FCCTexture()

        #all texture attributes are found in the FCC texture class
    def construct_SVE(self,fname1,fname2):
        # TODO: use the other functions to fill the rest of the attributes in this class
        return None

    #the first step is to build a bunch of orientation objects from the euler angles or quaternions
    #this can be done with the HCP or FCC texture classes
    def textureFromEulerAngles(self,fname,structure='FCC'):
        '''
        This is a function to create a texture object from euler angles using the Texture classes.

        :param fname: SVE feature data file. This is in the form of a csv.
        :param structure: The grain structure type of the SVE. This supports FCC and HCP through those subclasses.
        :return: None
        '''

        # Store number of grains in SVE
        df = pd.read_csv(fname,nrows=1,header=None)
        self.num_grains = df.iat[0, 0]
        self.sve_num = [char for char in fname if char.isdigit()]

        # Pull out euler angles
        df = pd.read_csv(fname, header=1, skiprows=0, nrows=self.num_grains)
        df = df.loc[:,['EulerAngles_0','EulerAngles_1','EulerAngles_2']]

        # Convert from csv to txt
        df.to_csv('EulerAngles.txt',header=None, index=None, sep=' ',mode='w+')

        # Build texture object from the file
        with open('EulerAngles.txt', 'r') as f:
            if structure == 'HCP':
                self.sveHCPTexture.fromEulerAnglesFile(f)
            elif structure == 'FCC':
                self.sveFCCTexture.fromEulerAnglesFile(f)

    #############################################
    def set_features(self,fname):
        '''
        This is a function to set the attributes of an SVE object that are found in the output feature data file
        each attribute is set as a dictionary with grain numbers as the keys.

        :param fname: SVE feature data file. This is in the form of a csv.
        :return: None
        '''

        # Store number of grains
        df = pd.read_csv(fname, nrows=1, header=None)
        self.num_grains = df.iat[0, 0]
        self.sve_num = [char for char in fname if char.isdigit()]

        # Read first third of csv
        feature_data = pd.read_csv(fname, header=1, skiprows=0, nrows=self.num_grains)
        feature_data.set_index('Feature_ID',inplace=True)

        # TODO: Cut out the 0 quaternions

        # Fill SVE attributes from first 1/3 of csv file
        for gnum in feature_data.index:
            # gnum = grain number
            # gname = grain name key
            gname = 'Grain_{}'

            self.quaternions[gname.format(gnum)] = feature_data.loc[gnum,['AvgQuats_0','AvgQuats_1','AvgQuats_2','AvgQuats_3']].tolist()
            self.axis_euler[gname.format(gnum)] = feature_data.loc[gnum,['AxisEulerAngles_0','AxisEulerAngles_1','AxisEulerAngles_2']].tolist()
            self.euler[gname.format(gnum)] = feature_data.loc[gnum,['EulerAngles_0','EulerAngles_1','EulerAngles_2']].tolist()
            self.centroid[gname.format(gnum)] = feature_data.loc[gnum,['Centroids_0','Centroids_1','Centroids_2']].tolist()
            self.volume[gname.format(gnum)] = feature_data.loc[gnum,'Volumes']
            self.omega3s[gname.format(gnum)] = feature_data.loc[gnum,'Omega3s'].tolist()
            self.axislengths[gname.format(gnum)] = feature_data.loc[gnum,['AxisLengths_0','AxisLengths_1','AxisLengths_2']].tolist()


    #############################################
    def set_grain_neighbors(self,fname):
        '''
        This function grabs the feature ID of the grains and their neighbors

        :param fname: SVE feature data file. This is in the form of a csv.
        :return: None
        '''

        # Store number of grains
        df = pd.read_csv(fname, nrows=1, header=None)
        self.num_grains = df.iat[0, 0]
        grains_and_neighbors = pd.read_csv(fname, skiprows=(self.num_grains+3), nrows=self.num_grains, header=None, sep='\n')

        # Fill attribute dictionary
        for grain in grains_and_neighbors.index:
            neighbors_cur = grains_and_neighbors.iloc[grain].to_string().split(',')
            del(neighbors_cur[0:2])
            self.grain_neighbor_link['Grain_{}'.format(grain + 1)] = list(map(int,neighbors_cur))

    #############################################
    def set_surface_area(self,fname):
        '''
        This function grabs the shared surface area between grains and their neighbors. It should be noted that
        the attribute dictionary is full of surface areas only and the labels of the neighboring grains should
        be pulled from the 'grain_neighbor_link' attribute.

        :param fname: SVE feature data file. This is in the form of a csv.
        :return: None
        '''

        # Store number of grains
        df = pd.read_csv(fname, nrows=1, header=None)
        self.num_grains = df.iat[0, 0]

        # Read shared surface area (ssa) portion of feature data
        ssa_data = pd.read_csv(fname, skiprows=((2 * self.num_grains) + 4), nrows=self.num_grains, header=None, sep='\n')

        # Fill attribute dictionary
        for grain in ssa_data.index:
            ssa_cur = ssa_data.iloc[grain].to_string().split(',')
            del(ssa_cur[0:2])
            self.shared_surface_area['Grain_{}'.format(grain + 1)] = list(map(float,ssa_cur))

        # label with grain neighbors
        for grain in self.shared_surface_area:
            neighbor_labels = {}
            for i,neighbor in enumerate(self.grain_neighbor_link[grain]):
                neighbor_labels[neighbor] = self.shared_surface_area[grain][i]
            self.neighbor_shared_surface_area[grain] = neighbor_labels

    #############################################
    def set_sub_band_data(self,fname):
        '''
        This is a function to retrieve the sub band max FIPs and store them in the
        appropriate attribute. Make sure num_grains is set before this

        :param fname: SVE feature data file. This is in the form of a csv.
        :return: None
        '''

        # Read data
        df = pd.read_csv(fname)
        df.set_index('grain',inplace=True)

        # Fill attribute dictionary
        for grain in df.index:
            self.max_fips['Grain_{}'.format(grain)] = df.loc[grain, 'FIP'].tolist()

        # # set element coordinates and FIPs
        # # TODO: Fill dictionary with elements in grain and their coords so distance from centroid can be calculated
        # # Make dataframe for ease of use
        # data = pd.read_csv(fname_ss,header=0)
        #
        # # Read shared surface area (ssa) portion of feature data
        # elem_data = pd.read_csv(fname_link, header=None, sep='\n')
        #
        # # Fill attribute dictionary
        # for grain in elem_data.index:
        #     elems_cur = elem_data.iloc[grain].to_string().split(',')
        #     elems_cur = [elem.strip() for elem in elems_cur]
        #     del(elems_cur[0])
        #     del(elems_cur[-1])
        #     self.grain_elem_link['Grain_{}'.format(grain + 1)] = list(map(int,elems_cur))

    #############################################
    def set_grain_element_data(self,fname):
        '''
        Link elements and their FIP with the corresponding grain

        :param fname
        :return: None
        '''
        # Grab elements and FIPs
        df = pd.read_csv(fname,usecols=['element','grain_id','FIPs'])
        # link and fill dictionary
        df = df.groupby('grain_id').agg({'element': list, 'FIPs': list})
        self.elem_grain_link = df.to_dict()


    #############################################
    def calc_misorientations(self,structure='FCC'):
        '''
        Calculate the maximum misorientations between all pairs of grains
        (neighboring or not) using the texture and grain classes. Then sort
        according to neighbors.
        :param structure: Grain structure of the SVE
        :return: None
        '''

        # Calculate the maximum misorientations from the texture class
        if structure == 'HCP':
            self.sveHCPTexture.calc_misorient()

            # Pair the actual neighbors and save
            for grain in self.sveHCPTexture.misorient:
                neighbor_labels = {}
                for neighbor in self.grain_neighbor_link[grain]:
                    neighbor_labels[neighbor] = self.sveHCPTexture.misorient[grain][neighbor - 1]
                self.neighbor_misorient[grain] = neighbor_labels

        elif structure == 'FCC':
            self.sveFCCTexture.calc_misorient()

            # Pair the actual neighbors and save
            for grain in self.sveFCCTexture.misorient:
                neighbor_labels = {}
                for neighbor in self.grain_neighbor_link[grain]:
                    neighbor_labels[neighbor] = self.sveFCCTexture.misorient[grain][neighbor - 1]
                self.neighbor_misorient[grain] = neighbor_labels

        # create pickle file for easy access.
        with open('neighbor_misorient.pkl', 'wb') as f:
            pickle.dump(self.neighbor_misorient,f)

    #############################################
    def calc_schmidFactors(self, structure='FCC',file_type='rod'):
        '''
        This function calculates the max Schmid factors (global) using the grain class
        the slip transmission (m') is also calculated here. A primary slip dictionary
        that stores the schmid factor, the normal to the plane, and the slip direction
        is created as well as a dictionary of m' values. A pickle file of m' values is
        also created for ease of use.

        :param fname: A desired file name. The file object is created here.
        :param structure: Grain structure of SVE
        :return: None
        '''

        # TODO: the rodrigues vectors are not being calculated correctly. It is causing a error in the Schmid factor calc
        # Open a file for max Schmid factors
        with open('schmidFactors.txt', 'w') as f:
            # Use texture class to calculate the Schmid factors
            # according to grain structure.

            if structure == 'HCP':
                if file_type == 'rod':
                    self.sveHCPTexture.toRodriguesFile2(f)
                elif file_type == 'euler':
                    self.sveHCPTexture.toEulerAnglesFile2(f)
            elif structure == 'FCC':
                if file_type == 'rod':
                    self.sveFCCTexture.toRodriguesFile2(f)
                elif file_type == 'euler':
                    self.sveFCCTexture.toEulerAnglesFile2(f)
    #############################################
    def calc_mPrime(self,structure_type='FCC',to_file=False):
        # determine texture
        if structure_type == 'HCP':
            texture = self.sveHCPTexture
        else:
            texture = self.sveFCCTexture

        # Calculate the maximum misorientations from the texture class
        texture.calc_mPrime()

        # Pair the actual neighbors and save
        for grain in texture.mp:
            neighbor_labels = {}
            try:
                for neighbor in self.grain_neighbor_link[grain]:
                    neighbor_labels[neighbor] = texture.mp[grain][neighbor - 1]
                    print(len(texture.mp[grain]))
                self.neighbor_mp[grain] = neighbor_labels
            except IndexError:
                print('Index Error @: ', neighbor - 1)

        # save to pickle file
        if to_file:
            with open('mprime.pkl', 'wb') as f:
                pickle.dump(self.neighbor_mp, f)

    #############################################
    def calc_fatigueLife(self,FIP):

        # TODO: find experimental values for IN625
        N = 0.0 # fatigue life of current band
        D_st = 0.0  # diameter of current band being evaluated
        phi = 0.0 # mechanical irreversibility at crack tip
        A = 0.0 # experimental constant
        b = 0.0 # experimental constant
        CTD = A*(FIP)**b # crack tip displacement

        #influence of neighboring grains
        n = None # number of neighboring bands. Calculate in CP-FE
        D_nd = None # diameter of neighboring bands
        theta_dis = None # angle of disorientation between two neighboring bands
        omega = 1 - theta_dis / 20 # disorientation factor
        influence_ng = [omega[neighbor] * D_nd[neighbor] for neighbor in n]

        # Beta term
        d_ref_gr = 0.0 # mean grain size of material (IN625)
        beta = (D_st + influence_ng) / d_ref_gr

        # constants
        c1 = phi * (beta * A * (FIP)**b - CTD)
        c2 = (phi * 2 * beta * A * (FIP)**b) / ((D_st + influence_ng)**2)

        # fatigue life of current band
        N.append((1 / np.sqrt(c1 * c2)) * np.arctanh(D_st * np.sqrt(c1/c2)))


        return None


    #############################################
    def calc_volumeAvg(self,fname_ss,fname_link):

        # Make dataframe for ease of use
        data = pd.read_csv(fname_ss,header=0)

        # Read shared surface area (ssa) portion of feature data
        elem_data = pd.read_csv(fname_link, header=None, sep='\n')

        # Fill attribute dictionary
        for grain in elem_data.index:
            elems_cur = elem_data.iloc[grain].to_string().split(',')
            elems_cur = [elem.strip() for elem in elems_cur]
            del(elems_cur[0])
            del(elems_cur[-1])
            self.grain_elem_link['Grain_{}'.format(grain + 1)] = list(map(int,elems_cur))

        # First build all of the matrices for each element

        # Max Peak

        strain_range=[]
        for grain in self.grain_elem_link:
            s_max = []
            e_max = []
            for elem in self.grain_elem_link[grain]:
                e_cur = np.empty((3, 3))
                s_cur = np.empty((3, 3))
                for j in range(0, 3):
                    if j == 1:
                        s_cur[j] = data.loc[elem+108000, ['S12', 'S' + str(j + 1) + '2', 'S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+108000, ['Ep21', 'Ep22', 'Ep23']]  # current stress matrix
                    elif j == 2:
                        s_cur[j] = data.loc[elem+108000, ['S13', 'S23', 'S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+108000, ['Ep31', 'Ep32', 'Ep33']]  # current strain matrix
                    else:
                        s_cur[j] = data.loc[elem+108000, ['S' + str(j + 1) + '1', 'S' + str(j + 1) + '2','S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+108000, ['Ep11', 'Ep12', 'Ep13']]  # current stress matrix
                e_max.append(e_cur)
                s_max.append(s_cur)

            s_min = []
            e_min = []
            for elem in self.grain_elem_link[grain]:
                e_cur = np.empty((3, 3))
                s_cur = np.empty((3, 3))
                for j in range(0, 3):
                    if j == 1:
                        s_cur[j] = data.loc[elem+81000, ['S12', 'S' + str(j + 1) + '2', 'S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+81000, ['Ep21', 'Ep22', 'Ep23']]  # current stress matrix
                    elif j == 2:
                        s_cur[j] = data.loc[elem+81000, ['S13', 'S23', 'S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+81000, ['Ep31', 'Ep32', 'Ep33']]  # current strain matrix
                    else:
                        s_cur[j] = data.loc[elem+81000, ['S' + str(j + 1) + '1', 'S' + str(j + 1) + '2','S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+81000, ['Ep11', 'Ep12', 'Ep13']]  # current stress matrix
                #e_min.append(max(np.linalg.eig(e_cur)[0], key=abs))  # max eigenvalue of strain in minimum
                e_min.append(e_cur)
                s_min.append(s_cur)  # max eigenvalue of stress in maximum

            sr = []
            for elem,strain in enumerate(e_max):
                sr.append(np.array(e_max[elem]) - np.array(e_min[elem]))
            self.strain_range[grain] = sr

            #self.grain_elem_stress[grain] = s
            #self.grain_elem_strain[grain] = e

            #self.vAvg_stress[grain] = np.mean(np.array(self.grain_elem_stress[grain]),axis=0)
            #self.vAvg_strain[grain] = strain_range

        return None

    #####################################
    def calc_fip_dist_from_boundary(self):

        # generate all element node coords
        node = 1
        # assume nodes start at [0,0,0] and increment by 2s.
        for z in range (0,51,2):
            for y in range (0,51,2):
                for x in range (0,51,2):
                    self.node_coords[node] = [x,y,z]
                    node += 1

        elem = 1
        # assuming element centroids start at [1,1,1] and go by 2s
        for z in range(1,50,2):
            for y in range(1,50,2):
                for x in range(1,50,2):
                    self.elem_coords[elem] = [x,y,z]
                    elem +=1            

        #build each element with node coords
        elem_nodes = {} # [[node1], [node2]....]
        val = list(self.node_coords.values())
        for elem,coords in self.elem_coords.items():
            cur = [] 
            #bottom nodes
            cur.append(val.index([coords[0]-1,coords[1]-1,coords[2]-1])+1)
            cur.append(val.index([coords[0]-1,coords[1]-1,coords[2]+1])+1)
            cur.append(val.index([coords[0]-1,coords[1]+1,coords[2]-1])+1)
            cur.append(val.index([coords[0]-1,coords[1]+1,coords[2]+1])+1)
            #top nodes
            cur.append(val.index([coords[0]+1,coords[1]+1,coords[2]+1])+1)
            cur.append(val.index([coords[0]+1,coords[1]+1,coords[2]+1])+1)
            cur.append(val.index([coords[0]+1,coords[1]+1,coords[2]+1])+1)
            cur.append(val.index([coords[0]+1,coords[1]+1,coords[2]+1])+1)
            elem_nodes[elem] = cur

        # loop through grains in SVE
        for grain in range(1,self.num_grains+1):
            grain_node_set = set()
            # find all node names in grain
            grain_node_list = [elem_nodes[elem] for elem in self.elem_grain_link['element'][grain]]
            grain_node_list = [item for sublist in grain_node_list for item in sublist] #flatten
            # update set
            grain_node_set.update(grain_node_list)
            # find all node names in all neighbors
            neighbors = [neighbor for neighbor in self.grain_neighbor_link['Grain_{}'.format(grain)]]
            #set for neighbor nodes
            neighbor_node_set = set()
            for neighbor in neighbors:
                neighbor_node_list = [elem_nodes[elem] for elem in self.elem_grain_link['element'][neighbor]]
                neighbor_node_list = [item for sublist in neighbor_node_list for item in sublist] #flatten
                neighbor_node_set.update(neighbor_node_list)
                
            self.grain_boundary_nodes['Grain_{}'.format(grain)] = grain_node_set.intersection(neighbor_node_set)

            # find element containing FIP
            #max_fip_elem = self.elem_grain_link['element'][argmax(self.elem_grain_link['FIPs'][grain])]
            # find coords of max FIP element
            #max_fip_elem_coords = self.elem_coords[max_fip_elem]
            # find minimum distance to boundary nodes
            #min_dist = np.linalg.norm()
    

        










