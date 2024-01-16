"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_stock
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list_stochastic

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

    root = "SBM_sub_vary"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    
    createFolder(fileName)

    ########################################################################################################
    #ONE SECTOR ONLY
    #NO carbon_price
    params["carbon_price_increased_lower"] = 0    
    params_list_no_tau = produce_param_list_stochastic(params, property_values_list, property_varied)

    #Low carbon price
    params["carbon_price_increased_lower"] = 0.1   
    params_list_low_tau = produce_param_list_stochastic(params, property_values_list, property_varied)

    #High carbon rpice
    params["carbon_price_increased_lower"] = 0.5   
    params_list_high_tau = produce_param_list_stochastic(params, property_values_list, property_varied)

    params_list = params_list_no_tau + params_list_low_tau + params_list_high_tau

    ########################################################################################################
    #Both SECTORS
    var_params["block_heterogenous_sector_substitutabilities_state"] = 0 #EVERYONE HAS THE SAME SUBSTITUTABILITES FOR BOTH SECTORS AND BLOCKS, THIS IS THE REFERENCE CASE FOR WHAT THE EFFECT OF INCREASING SUBSTITUTABILITIES IS
    var_params["property_varied"] = "low_carbon_substitutability_upper"
    property_varied = var_params["property_varied"]
    property_values_list = generate_vals(
        var_params
    )

    #NO carbon_price
    params["carbon_price_increased_lower"] = 0    
    params_list_no_tau_both = produce_param_list_stochastic(params, property_values_list, property_varied)

    #Low carbon price
    params["carbon_price_increased_lower"] = 0.1   
    params_list_low_tau_both = produce_param_list_stochastic(params, property_values_list, property_varied)

    #High carbon rpice
    params["carbon_price_increased_lower"] = 0.5   
    params_list_high_tau_both = produce_param_list_stochastic(params, property_values_list, property_varied)

    a= [b[""] for b in params_list_high_tau_both]
    print()

    params_list_both = params_list_no_tau_both + params_list_low_tau_both + params_list_high_tau_both
    print("TOTAL RUNS", len(params_list)+ len(params_list_both))
    
    emissions_stock_serial = multi_emissions_stock(params_list)
    emissions_array = emissions_stock_serial.reshape(3, property_reps, params["seed_reps"])#3 is for the 3 differents states

    emissions_stock_serial_both = multi_emissions_stock(params_list_both)
    emissions_array_both = emissions_stock_serial_both.reshape(3, property_reps, params["seed_reps"])#3 is for the 3 differents states

    save_object(emissions_array, fileName + "/Data", "emissions_array")
    save_object(emissions_array_both, fileName + "/Data", "emissions_array_both")
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_SBM_sub.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_SBM_sub.json",
        RUN_TYPE = 5
)

