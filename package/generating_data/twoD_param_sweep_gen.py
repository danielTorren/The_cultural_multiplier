"""Run multiple simulations varying two parameters
A module that use input data to generate data from multiple social networks varying two properties
between simulations so that the differences may be compared. Useful for comparing the influences of these parameters
on each other to generate phase diagrams.



Created: 10/10/2022
"""
# imports
import json
import numpy as np
from logging import raiseExceptions
from matplotlib.colors import Normalize, LogNorm
from package.resources.utility import (
    createFolder,
    save_object,
    produce_name_datetime,
)
from package.resources.run import (
    parallel_run_sa,
)

# modules
def produce_param_list_n_double(
    params_dict: dict, variable_parameters_dict: dict[dict]
) -> list[dict]:
    """Creates a list of the param dictionaries. This only varies both parameters at the same time in a grid like fashion.

    Parameters
    ----------
    params_dict: dict,
        dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters.
    variable_parameters_dict: dict[dict]
        dictionary of dictionaries containing details for range of parameters to vary.

    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []

    for i in variable_parameters_dict["row"]["vals"]:
        for j in variable_parameters_dict["col"]["vals"]:
            params_dict[variable_parameters_dict["row"]["property"]] = i
            params_dict[variable_parameters_dict["col"]["property"]] = j
            params_list.append(params_dict.copy())

    return params_list

def generate_vals_variable_parameters_and_norms(variable_parameters_dict):
    """using minimum and maximum values for the variation of a parameter generate a list of
     data and what type of distribution it uses

     Parameters
    ----------
    variable_parameters_dict: dict[dict]
        dictionary of dictionaries  with parameters used to generate attributes, dict used for readability instead of super
        long list of input parameters. Each key in this out dictionary gives the names of the parameter to be varied with details
        of the range and type of distribution of these values found in the value dictionary of each entry.

    Returns
    -------
    variable_parameters_dict: dict[dict]
        Same dictionary but now with extra entries of "vals" and "norm" in the subset dictionaries

    """
    for i in variable_parameters_dict.values():
        if i["divisions"] == "linear":
            i["vals"] = np.linspace(i["min"], i["max"], i["reps"])
            i["norm"] = Normalize()
        elif i["divisions"] == "log":
            i["vals"] = np.logspace(i["min"], i["max"], i["reps"])
            i["norm"] = LogNorm()
        else:
            raiseExceptions("Invalid divisions, try linear or log")
    return variable_parameters_dict

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_2D.json"
    ) -> str: 

    # load base params
    f_base_params = open(BASE_PARAMS_LOAD)
    base_params = json.load(f_base_params)
    f_base_params.close()

    # load variable params
    f_variable_parameters = open(VARIABLE_PARAMS_LOAD)
    variable_parameters_dict = json.load(f_variable_parameters)
    f_variable_parameters.close()

    # AVERAGE OVER MULTIPLE RUNS
    variable_parameters_dict = generate_vals_variable_parameters_and_norms(
        variable_parameters_dict
    )

    root = "two_param_sweep_average"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    params_list = produce_param_list_n_double(base_params, variable_parameters_dict)
    (
        results_emissions_stock,
        results_emissions_flow,
        results_mu,
        results_var,
        results_coefficient_of_variance,
        results_emissions_change
    ) = parallel_run_sa(params_list)

    createFolder(fileName)

    save_object(base_params, fileName + "/Data", "base_params")
    # save the data and params_list
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    save_object(results_emissions_stock, fileName + "/Data", "results_emissions_stock")
    save_object(results_emissions_flow, fileName + "/Data", "results_emissions_flow")
    save_object(results_mu, fileName + "/Data", "results_mu")
    save_object(results_var, fileName + "/Data", "results_var")
    save_object(results_coefficient_of_variance,fileName + "/Data","results_coefficient_of_variance")
    save_object(results_emissions_change, fileName + "/Data", "results_emissions_change")

    return fileName

if __name__ == '__main__':
    fileName_Figure_11 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_2D_B_d.json"
    )