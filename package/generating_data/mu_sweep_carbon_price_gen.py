"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_stochstic_emissions_flow_stock_run,multi_emissions_flow_stock_run

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

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
        CARBON_PRICES_LOAD = "package/carbon_prices_mu_sweep.json"
         ) -> str: 

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
    property_min = var_params["property_min"]#0,
    property_max = var_params["property_max"]#1,
    property_reps = var_params["property_reps"]#10,
    property_varied_title = var_params["property_varied_title"]# #"A to Omega ratio"
    property_values_list = np.linspace(property_min, property_max, property_reps)


    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    time_array = np.arange(0,params["time_steps_max"] + params["compression_factor_state"],params["compression_factor_state"])

    f_carbon_prices = open(CARBON_PRICES_LOAD)
    carbon_prices = json.load(f_carbon_prices)

    root = "mu_sweep_carbon_price"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    stocH_params_list = []
    for price in carbon_prices["carbon_prices"]:
        params["carbon_price_increased"] = price
        stocH_params_list = stocH_params_list  + produce_param_list_stochastic(params, property_values_list, property_varied)

    emissions_stock_array, emissions_flow_array, identity_array = multi_emissions_flow_stock_run(stocH_params_list)

    emissions_stock_data_list = emissions_stock_array.reshape((len(carbon_prices["carbon_prices"]),property_reps, params["seed_reps"],len(time_array)))
    emissions_flow_data_list = emissions_flow_array.reshape((len(carbon_prices["carbon_prices"]),property_reps, params["seed_reps"],len(time_array)))
    identity_array_data_list = identity_array.reshape((len(carbon_prices["carbon_prices"]),property_reps, params["seed_reps"],len(time_array),params["N"]))
    
    createFolder(fileName)

    save_object(emissions_stock_data_list, fileName + "/Data", "emissions_stock_data_list")
    save_object(emissions_flow_data_list, fileName + "/Data", "emissions_flow_data_list")
    save_object(identity_array_data_list,fileName + "/Data", "identity_array_data_list")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")
    save_object(carbon_prices, fileName + "/Data", "carbon_prices")
    save_object(time_array, fileName + "/Data", "time_array")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_mu_sweep.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_mu.json",
        CARBON_PRICES_LOAD = "package/constants/carbon_prices_mu_sweep.json"

)

