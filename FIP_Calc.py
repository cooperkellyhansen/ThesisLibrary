"""""

Calculate some FIPs
This data is just for Sample1 and the 0th instantiation

Im thinking that a FIP could be an object with some sort of attributes



"""""
import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt




def FIP_Calculator(data):
    # takes in a dataframe of FIP data

    # globals
    k = 1.0
    macro_yield = 830.0 #MPa

    #solve for eigenvalues of stress and strain. Each row is the state of stress of a single element.
    e_max = [] #list of max strain vals
    s_max = [] #list of max stress vals
    e_min = [] #list of min strain vals
    s_min = [] #list of min stress vals
    strain_range = [] #list of strain range vals
    max_normal_stress = [] #maximum normal stress
    FIPs = []
    e_cur = np.empty((3,3))
    s_cur = np.empty((3,3))
    #We will be taking from the bottom of the csv
    #which is the 3rd cycle of loading. This will be the MAX of the FINAL MAX peak stress/strain
    #build matrix for eigenvalue calculation
    for i in range(134999, 162000):
        for j in range(0,3):
            if j==1:
                s_cur[j] = data.loc[i, ['S12', 'S' + str(j + 1) + '2', 'S' + str(j + 1) + '3']]  # current stress matrix
                e_cur[j] = data.loc[i, ['Ep21', 'Ep22', 'Ep23']]  # current stress matrix
            elif j==2:
                s_cur[j] = data.loc[i, ['S13', 'S23', 'S' + str(j + 1) + '3']]  # current stress matrix
                e_cur[j] = data.loc[i, ['Ep31', 'Ep32', 'Ep33']]  # current strain matrix
            else:
                s_cur[j] = data.loc[i, ['S' + str(j + 1) + '1', 'S' + str(j + 1) + '2', 'S' + str(j + 1) + '3']] #current stress matrix
                e_cur[j] = data.loc[i, ['Ep11', 'Ep12','Ep13']]  # current stress matrix
        e_max.append(max(np.linalg.eig(e_cur)[0], key=abs)) # abs max eigenvalue of strain in max final peak
        s_max.append(max(np.linalg.eig(s_cur)[0], key=abs)) # abs max eigenvalue of stress in max final peak

    #This will be the MAX of the FINAL MIN valley stress/strain
    #build matrix for eigenvalue calculation
    for i in range(108001, 135002):
        for j in range(0,3):
            if j==1:
                s_cur[j] = data.loc[i, ['S12', 'S' + str(j + 1) + '2', 'S' + str(j + 1) + '3']]  # current stress matrix
                e_cur[j] = data.loc[i, ['Ep21', 'Ep22', 'Ep23']]  # current stress matrix
            elif j==2:
                s_cur[j] = data.loc[i, ['S13', 'S23', 'S' + str(j + 1) + '3']]  # current stress matrix
                e_cur[j] = data.loc[i, ['Ep31', 'Ep32', 'Ep33']]  # current strain matrix
            else:
                s_cur[j] = data.loc[i, ['S' + str(j + 1) + '1', 'S' + str(j + 1) + '2', 'S' + str(j + 1) + '3']] #current stress matrix
                e_cur[j] = data.loc[i, ['Ep11', 'Ep12','Ep13']]  # current stress matrix
        e_min.append(max(np.linalg.eig(e_cur)[0],key=abs)) # max eigenvalue of strain in minimum
        s_min.append(max(np.linalg.eig(s_cur)[0],key=abs)) # max eigenvalue of stress in maximum

    #Now that we have the principal angles we need to rotate to the slip plane


    #loop through and calculate the strain range and max normal stress
    for i in range(len(e_max)):
        strain_range.append(abs(e_max[i] - e_min[i]))
        max_normal_stress.append(max([s_min[i],s_max[i]], key=abs))

    #calculate FIP for each element
    for i in range(len(max_normal_stress)):
        FIPs.append((strain_range[i]/2.0)*(1.0+(k*max_normal_stress[i]/macro_yield)))
    print('e_max', e_max)
    FIP_data = pd.DataFrame(FIPs)
    FIP_data.to_csv('c:/Users/coope/PycharmProjects/pythonProject/sample_1/Results_Ep_0_FIP.csv')

    return FIPs,strain_range


#takes in FIPs list
def plot_FIP(FIPs):
    #plot a histogram of the FIPs
    plt.figure()
    plt.hist(FIPs, bins=20)
    plt.ylabel('f(x)')
    plt.xlabel('FIP')
    plt.title('FIPs - Sample 0 - SVE 0')
    #plt.xscale('loglog')
    #plt.yscale('log')
    plt.show()


def FIP_link(data_link, FIPs):
    #split elements into respective grains (16000)
    elem_grain_dict = {} #element grain link
    max_FIP_dict = {} #max FIP in each
    data_link_clean = [] # clean data link list
    for i in range(len(data_link)): #loop thru rows (grains)
        grain_name = 'Grain_{}'.format(i+1)  # create grain name
        data_cur = []
        link_cur = data_link.loc[i].dropna().astype('str').tolist() #drop nan terms cast to a list of strings
        #loop to clean data
        for k in range(len(link_cur)):
            link_cur[k] = link_cur[k].strip() #remove any spaces
            if '' in link_cur:
                link_cur.remove('') #remove any blank terms
        int_map = map(int,link_cur) #cast back to ints
        link_cur = list(int_map)
        data_link_clean.append(link_cur)
        max_FIP = FIPs[link_cur[0]-1]
        for j in range(1,len(link_cur)):   #loop thru columns of current row (elements in grain)
            data_cur.append(FIPs[link_cur[j]-1])   #build data_cur with FIP vals of a single grain
            if max_FIP < FIPs[link_cur[j]-1]:   #keep track of max
                max_FIP = FIPs[link_cur[j]-1]
        elem_grain_dict[grain_name] = data_cur   #fill dict with grain labels and corresponding FIPs
        #print(elem_grain_dict)
        # capture top FIP in each grain (16000)
        max_FIP_dict[grain_name] = max_FIP

    FIP_list = list(elem_grain_dict.values())  #find mean of each grains FIPs
    avg_FIP = []
    for i in FIP_list:
        avg_FIP.append(sum(i)/len(i))

    return elem_grain_dict,max_FIP_dict,avg_FIP,data_link_clean

#pull top 10 percent and grain labels (160)