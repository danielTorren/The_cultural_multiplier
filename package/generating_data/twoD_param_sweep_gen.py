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
    multi_emissions_stock, emissions_parallel_run
)

# modules
def produce_param_list_stochastic_n_double(params_dict: dict, variable_parameters_dict: dict[dict]) -> list[dict]:
    params_list = []
    key_param_array = []
    for i in variable_parameters_dict["row"]["vals"]:
        key_param_row = []
        for j in variable_parameters_dict["col"]["vals"]:
            params_dict[variable_parameters_dict["row"]["property"]] = i
            params_dict[variable_parameters_dict["col"]["property"]] = j
            key_param_row.append((i,j))
            for v in range(params_dict["seed_reps"]):
                params_dict["set_seed"] = int(v+1)#as 0 and 1 are the same seed
                params_list.append(params_dict.copy())
        key_param_array.append(key_param_row)
 
    return params_list,key_param_array

def generate_vals_variable_parameters_and_norms(variable_parameters_dict):

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

    root = "two_param_sweep"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    params_list,key_param_array = produce_param_list_stochastic_n_double(base_params, variable_parameters_dict)
    print("TOTAL RUNS: ", len(params_list))
    results_emissions_stock_series = emissions_parallel_run(params_list)

    results_emissions_stock = results_emissions_stock_series.reshape((variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"], base_params["seed_reps"]))

    createFolder(fileName)

    save_object(params_list, fileName + "/Data", "params_list")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    save_object(results_emissions_stock, fileName + "/Data", "results_emissions_stock")
    save_object(key_param_array, fileName + "/Data","key_param_array")

    return fileName

if __name__ == '__main__':
    fileName_Figure_11 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_2D_sigma_sigma.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_2D_sigma_sigma.json"
    )