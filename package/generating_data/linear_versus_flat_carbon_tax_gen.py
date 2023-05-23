"""
Created: 10/10/2022
"""

# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_stochstic_emissions_run,multi_emissions_stock
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list_stochastic

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        CARBON_PRICES_LOAD = "package/carbon_prices.json"
         ) -> str: 

    f_var = open(CARBON_PRICES_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property"]
    property_min = var_params["min"]
    property_max = var_params["max"]
    property_reps = var_params["reps"]

    property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "flat_linear_sweep_carbon_price"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    params["carbon_tax_implementation"] = "flat"
    params_list_flat = produce_param_list_stochastic(params, property_values_list, property_varied)#produce_param_list(params, property_values_list,property_varied)
    params["carbon_tax_implementation"] = "linear"
    params_list_linear = produce_param_list_stochastic(params, property_values_list, property_varied)#produce_param_list(params, property_values_list, property_varied)

    stocH_params_list = params_list_flat + params_list_linear
    #print(len(stocH_params_list), len(params_list_flat), len(params_list_linear))

    emissions_stock_array = multi_emissions_stock(stocH_params_list)

    emissions_stock_data_list = emissions_stock_array.reshape(2,property_reps, params["seed_reps"])

    data_flat = emissions_stock_data_list[0] 
    data_linear = emissions_stock_data_list[1]

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
        BASE_PARAMS_LOAD = "package/constants/base_params_carbon_prices_linear_versus_flat.json",
        CARBON_PRICES_LOAD = "package/constants/carbon_prices_linear_versus_flat.json"
)

