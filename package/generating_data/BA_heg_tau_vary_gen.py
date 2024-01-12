"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_stock, multi_emissions_stock_ineq,parallel_run
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list_stochastic,produce_param_list
from package.generating_data.twoD_param_sweep_gen import generate_vals_variable_parameters_and_norms

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(np.log10(variable_parameters_dict["property_min"]),np.log10( variable_parameters_dict["property_max"]), variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_values_list 

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
        RUN_TYPE = 1
         ) -> str: 

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
    property_reps = var_params["property_reps"]#10,

    property_values_list = generate_vals(
        var_params
    )
    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "BA_heg_tau_vary"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    
    createFolder(fileName)

    #NO HEGEMONOY AND COMPLETE MIXING
    params["BA_brown_or_green_hegemony"] = 0    
    params["homophily_state"] = 0
    params_list_no_heg = produce_param_list_stochastic(params, property_values_list, property_varied)

    #Green HEGEMONOY AND homophily
    params["BA_brown_or_green_hegemony"] = -1   
    params["homophily_state"] = 1
    params_list_green_heg = produce_param_list_stochastic(params, property_values_list, property_varied)

    #Brown HEGEMONOY AND homophily
    params["BA_brown_or_green_hegemony"] = 1   
    params["homophily_state"] = 1
    params_list_brown_heg = produce_param_list_stochastic(params, property_values_list, property_varied)

    params_list = params_list_no_heg + params_list_green_heg + params_list_brown_heg
    print("TOTAL RUNS", len(params_list))
    
    emissions_stock_serial = multi_emissions_stock(params_list)
    print(emissions_stock_serial.shape)
    emissions_array = emissions_stock_serial.reshape(3, property_reps, params["seed_reps"])#3 is for the 3 differents states
    print(emissions_array.shape)

    save_object(emissions_array, fileName + "/Data", "emissions_array")
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_BA_heg_tau.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_BA_heg_tau.json",
        RUN_TYPE = 5
)

