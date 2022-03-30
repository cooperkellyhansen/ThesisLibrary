"""""
Author: Cooper Hansen
Running bingo on some data
Data: Gary Whelan PhD

This class separates the data from the CP-FEM
simulations and prepares it as training data
for dimensional reduction. It utilizes specifically
the Whelan data. 


"""""

import pandas as pd
import os

"""""
separate each set of FIPS on a given sample

Arguments:
sample_num = an integer corresponding to the desired sample number
desired_columns = string list of desired columns to extract
column names: [FIP,Grain,Slip,LayerBand,SubBand,SVE,Iteration]
* Slip = slip system
* LayerBand = the layer band within the slip system
* SubBand = the sub band within the layer band

Return: a dictionary of sub band data 
"""""


def sub_band_data(sample_num, desired_columns):
    # read csv data
    data = pd.read_csv('c:/Users/coope/PycharmProjects/pythonProject/sample_' + str(
        sample_num) + '/sub_band_averaged_max_per_grain.csv', header=None)
    # name columns. They are not named in csv
    data.columns = ['FIP', 'Grain', 'Slip', 'LayerBand', 'SubBand', 'SVE', 'Iteration']
    sub_band_dict = {}
    for i in range(0, 40):
        SVE_num = i
        sub_band_cur = data.loc[data['SVE'] == SVE_num]  # select rows with corresponding SVE number and order them
        SVE_name = 'SVE_{}'.format(SVE_num)  # set SVE name in dictionary
        sub_band_cur = sub_band_cur.loc[:, desired_columns]  # reduce current dictionary to only desired columns
        sub_band_dict[SVE_name] = sub_band_cur  # set FIP values for corresponding SVE
    return sub_band_dict


# separate each set of euler angles on a given sample
# sample_num = the number of the sample that the data should be pulled from
# desired_columns = string list of desired columns
# need to document column names
def feature_data(sample_num, desired_columns):
    # dictionary for feature data
    feature_data_dict = {}
    # read data for each feature csv
    for i in range(0, 40):
        # read first line of each csv to find number of grains
        feature_data = pd.read_csv(
            'c:/Users/coope/PycharmProjects/pythonProject/sample_' + str(sample_num) + '/FeatureData_FakeMatl_' + str(
                i) + '.csv', nrows=1, header=None)
        # pull out number of grains in current SVE
        num_grains = feature_data.iat[0, 0]
        # read the rest of the csv up until it splits
        feature_data = pd.read_csv(
            'c:/Users/coope/PycharmProjects/pythonProject/sample_' + str(sample_num) + '/FeatureData_FakeMatl_' + str(
                i) + '.csv', header=1, skiprows=0, nrows=num_grains)
        # pull out desired columns
        data_cur = feature_data.loc[:, desired_columns]
        # assign desired columns to the corresponding SVE number
        SVE_num = i
        SVE_name = 'SVE_{}'.format(SVE_num)  # set SVE name in dictionary
        feature_data_dict[SVE_name] = data_cur  # set vals for corresponding SVE
    return feature_data_dict


# def feature_data_neighbor(sample_num):
#     # create dictionary for neighbor data in csv
#     feature_neighbor_data_dict = {}
#     # assign desired columns to the corresponding SVE number
#     for i in range(0,40): # loop through number of SVE's in ensemble
#         feature_data = pd.read_csv('c:/Users/coope/PycharmProjects/pythonProject/sample_' + str(sample_num) + '/FeatureData_FakeMatl_' + str(i) + '.csv',nrows=1, header=None)
#         # pull out number of grains in current SVE
#         num_grains = feature_data.iat[0,0]
#         # read the neighbor portion of the csv
#         column_names = [i for i in range(0, 203)] #must name columns but I need to find largest number of columns
#         feature_data_neighbor_df = pd.read_csv('c:/Users/coope/PycharmProjects/pythonProject/sample_' + str(sample_num) + '/FeatureData_FakeMatl_' + str(i) + '.csv', header=(num_grains + 3), skiprows=(num_grains+2), nrows=num_grains, dtype=object)
#         #clean data and get list of neighbors
#         data_cur = feature_data_neighbor_df.loc[:,:]
#         # assign desired columns to the corresponding SVE number
#         SVE_num = i
#         SVE_name = 'SVE_{}'.format(SVE_num)  # set SVE name in dictionary
#         feature_neighbor_data_dict[SVE_name] = data_cur  # set vals for corresponding SVE
#     return feature_neighbor_data_dict


def feature_surface_area_data(sample_num):
    # dictionary for surface area data in csv
    feature_data_surface_area_dict = {}
    # read the surface area portion of the csv
    for i in range(0, 40):
        # read first line of each csv to find number of grains
        feature_data = pd.read_csv(
            'c:/Users/coope/PycharmProjects/pythonProject/sample_' + str(sample_num) + '/FeatureData_FakeMatl_' + str(
                i) + '.csv', nrows=1, header=None)
        # pull out number of grains in current SVE
        num_grains = feature_data.iat[0, 0]
        feature_data_area = pd.read_csv(
            'c:/Users/coope/PycharmProjects/pythonProject/sample_' + str(sample_num) + '/FeatureData_FakeMatl_' + str(
                i) + '.csv', header=((2 * num_grains) + 3), names=[i for i in range(0, 28)])
        # assign desired columns to the corresponding SVE number
        SVE_num = i
        SVE_name = 'SVE_{}'.format(SVE_num)  # set SVE name in dictionary
        feature_data_surface_area_dict[SVE_name] = feature_data_area

    return feature_data_surface_area_dict


# separate stress and strain data
def stress_strain_data(sample_num):
    stress_strain_data_dict = {}
    # read csv data
    for i in range(0, 40):
        # find the file name using the first part of the string
        for file in os.listdir('c:/Users/coope/PycharmProjects/pythonProject/sample_' + str(sample_num)):
            file_str = 'Results_main_' + str(i)
            if file.startswith(file_str):
                file_name = file
        stress_strain_data = pd.read_csv(
            'c:/Users/coope/PycharmProjects/pythonProject/sample_' + str(sample_num) + '/' + file_name)
        # pull out desired columns
        data_cur = stress_strain_data.loc[:, :]
        # assign desired columns to the corresponding SVE number
        SVE_num = i
        SVE_name = 'SVE_{}'.format(SVE_num)  # set SVE name in dictionary
        stress_strain_data_dict[SVE_name] = data_cur  # set vals for corresponding SVE

    return stress_strain_data_dict


def convert_to_dataframe(dict):
    return True


def quaternion_neighbor_data(quaternion_data_link, quaternion_data):
    quat_grain_and_neighbor_dict = {}
    # each grain will now have its own quaternions and its neighbors quaternions
    for i in range(len(quaternion_data_link)):  # loop thru rows (grains)
        grain_name = 'Grain_{}'.format(i + 1)  # create grain name
        data_cur = [quaternion_data[i]]  # initialize as list of lists
        link_cur = quaternion_data_link.loc[i].dropna().drop([0, 1]).astype(
            'str').tolist()  # drop nan terms cast to a list of strings. link_cur is a list of the current grains neighbors
        # loop to clean data
        for k in range(len(link_cur)):
            link_cur[k] = link_cur[k].strip()  # remove any spaces
            if '' in link_cur:
                link_cur.remove('')  # remove any blank terms
        float_map = map(float, link_cur)  # cast back to ints
        link_cur = list(float_map)
        int_map = map(int, link_cur)
        link_cur = list(int_map)
        # use link_cur values
        for j in range(0, len(link_cur)):  # loop thru columns of current row (grain, neighbors of grain)
            data_cur.append(
                quaternion_data[link_cur[j] - 1])  # build data_cur with quaternions of grain and its neighbors
        quat_grain_and_neighbor_dict[grain_name] = data_cur  # fill dict with grain labels and corresponding quats
    return quat_grain_and_neighbor_dict


