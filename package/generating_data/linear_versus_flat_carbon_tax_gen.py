"""
Created: 10/10/2022
"""

# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_stochstic_emissions_run

# modules
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

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        property_min = 0,
        property_max = 2,
        property_reps = 16,
         ) -> str: 


    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "linear_versus_flat_carbon_tax"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    
    property_values_list = np.linspace(property_min, property_max, property_reps)

    params["carbon_tax_implementation"] = "flat"
    params_list_flat = produce_param_list(params, property_values_list, "carbon_price_increased")
    params["carbon_tax_implementation"] = "linear"
    params_list_linear = produce_param_list(params, property_values_list, "carbon_price_increased")

    data_flat = multi_stochstic_emissions_run(params_list_flat) 
    data_linear = multi_stochstic_emissions_run(params_list_linear) 

    createFolder(fileName)

    save_object(data_flat, fileName + "/Data", "data_flat")
    save_object(data_linear, fileName + "/Data", "data_linear")
    save_object(params_list_flat, fileName + "/Data", "params_list_flat")
    save_object(params_list_linear, fileName + "/Data", "params_list_linear")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_values_list, fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        property_min = 0,
        property_max = 2,
        property_reps = 16,
)

