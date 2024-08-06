"""Contains functions that are not crucial to the simulation itself and are shared amongst files.
A module that aides in preparing folders, saving, loading and generating data for plots.

Created: 10/10/2022
"""

# imports
import pickle
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy import stats

# modules
def produce_name_datetime(root):
    fileName = "results/" + root +  "_" + datetime.datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    return fileName

def check_other_folder():
        # make prints folder:
    plotsName = "results/Other"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)
        
def get_cmap_colours(n, name='plasma'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

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

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(np.log10(variable_parameters_dict["property_min"]),np.log10( variable_parameters_dict["property_max"]), variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"]== "geo":
        property_values_list = np.geomspace(variable_parameters_dict["property_min"],variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_values_list 

def produce_param_list(params: dict, property_list: list, property: str) -> list[dict]:
    """
    Produce a list of the dicts for each experiment

    Parameters
    ----------
    params: dict
        base parameters from which we vary e.g
            params["time_steps_max"] = int(params["total_time"] / params["delta_t"])
    porperty_list: list
        list of values for the property to be varied
    property: str
        property to be varied

    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []
    for i in property_list:
        params[property] = i
        params_list.append(
            params.copy()
        )  # have to make a copy so that it actually appends a new dict and not just the location of the params dict
    return params_list

def calc_bounds(data, confidence_level):
    # Calculate mean and standard deviation across rows
    ys_mean = data.mean(axis=1)
    ys_std = data.std(axis=1)

    # Calculate the standard error of the mean (SEM)
    n = data.shape[1]  # Number of samples
    ys_sem = ys_std / np.sqrt(n)

    # Calculate margin of error
    z_score = np.abs(stats.norm.ppf((1 - confidence_level) / 2))  # For a two-tailed test
    margin_of_error = z_score * ys_sem

    # Calculate confidence intervals
    lower_bound = ys_mean - margin_of_error
    upper_bound = ys_mean + margin_of_error

    return ys_mean,lower_bound, upper_bound

def produce_param_list_stochastic(params: dict, property_list: list, property: str) -> list[dict]:
    params_list = []
    for i in property_list:
        params[property] = i
        for v in range(params["seed_reps"]):
            params["set_seed"] = int(v+1)
            params_list.append(
                params.copy()
            )  
    return params_list

def produce_param_list_only_stochastic(params: dict) -> list[dict]:
    params_list = []
    for v in range(params["seed_reps"]):
        params["set_seed"] = int(v+1)
        params_list.append(
            params.copy()
        )  
    return params_list

def produce_param_list_stochastic_multi(params: dict, property_list: list, property: str) -> list[dict]:
    seeds_labels = ["preferences_seed", "network_structure_seed", "shuffle_homophily_seed", "shuffle_coherance_seed"]
    params_list = []
    for i in property_list:
        params[property] = i
        for j in range(params["seed_reps"]):
            for k, label in enumerate(seeds_labels):
                params[label] = int(10*k + j + 1)
                params_list.append(
                    params.copy()
                )  
    return params_list

def produce_param_list_only_stochastic_multi(params: dict) -> list[dict]:
    seeds_labels = ["preferences_seed", "network_structure_seed", "shuffle_homophily_seed", "shuffle_coherance_seed"]
    params_list = []
    for j in range(params["seed_reps"]):
        for k, label in enumerate(seeds_labels):
            params[label] = int(10*k + j + 1)
            params_list.append(
                params.copy()
            )  
    return params_list