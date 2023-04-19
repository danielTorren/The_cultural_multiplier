"""Contains functions that are not crucial to the simulation itself and are shared amongst files.
A module that aides in preparing folders, saving, loading and generating data for plots.

Created: 10/10/2022
"""

# imports
import pickle
import os
import numpy as np
from scipy.signal import argrelextrema
import datetime
from sklearn.neighbors import KernelDensity

# modules
def produce_name_datetime(root):
    fileName = "results/" + root +  "_" + datetime.datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    return fileName

def createFolder(fileName: str) -> str:
    """
    Check if folders exist and if they dont create results folder in which place Data, Plots, Animations
    and Prints folders

    Parameters
    ----------
    fileName:
        name of file where results may be found

    Returns
    -------
    None
    """

    # print(fileName)
    # check for resutls folder
    if str(os.path.exists("results")) == "False":
        os.mkdir("results")

    # check for runName folder
    if str(os.path.exists(fileName)) == "False":
        os.mkdir(fileName)

    # make data folder:#
    dataName = fileName + "/Data"
    if str(os.path.exists(dataName)) == "False":
        os.mkdir(dataName)
    # make plots folder:
    plotsName = fileName + "/Plots"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    # make animation folder:
    plotsName = fileName + "/Animations"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    # make prints folder:
    plotsName = fileName + "/Prints"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

def save_object(data, fileName, objectName):
    """save single object as a pickle object

    Parameters
    ----------
    data: object,
        object to be saved
    fileName: str
        where to save it e.g in the results folder in data or plots folder
    objectName: str
        what name to give the saved object

    Returns
    -------
    None
    """
    with open(fileName + "/" + objectName + ".pkl", "wb") as f:
        pickle.dump(data, f)

def load_object(fileName, objectName) -> dict:
    """load single pickle file

    Parameters
    ----------
    fileName: str
        where to load it from e.g in the results folder in data folder
    objectName: str
        what name of the object to load is

    Returns
    -------
    data: object
        the pickle file loaded
    """
    with open(fileName + "/" + objectName + ".pkl", "rb") as f:
        data = pickle.load(f)
    return data

def calc_pos_clusters_set_bandwidth(identity_data,s,bandwidth):
    kde, e = calc_num_clusters_set_bandwidth(identity_data,s,bandwidth)
    ma = argrelextrema(e, np.greater)[0]
    list_identity_clusters = s[ma]
    return list_identity_clusters

def calc_num_clusters_set_bandwidth(identity_data,s,bandwidth):
    X_reshape = identity_data.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_reshape)
    e = kde.score_samples(s.reshape(-1,1))
    return kde, e
    
def get_cluster_list(identity_data,s, N, mi):

    index_list = np.arange(N)

    #left edge
    left_mask = (identity_data < s[mi][0])
    clusters_index_lists = [list(index_list[left_mask])]

    #  all middle cluster
    for i_cluster in range(len(mi)-1):
        center_mask = ((identity_data >= s[mi][i_cluster])*(identity_data <= s[mi][i_cluster+1]))
        clusters_index_lists.append(list(index_list[center_mask]))

    # most right cluster
    right_mask = (identity_data >= s[mi][-1])
    clusters_index_lists.append(list(index_list[right_mask]))

    return clusters_index_lists