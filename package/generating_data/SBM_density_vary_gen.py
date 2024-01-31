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
from package.resources.utility import generate_vals

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

    root = "SBM_density_vary"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    
    createFolder(fileName)

    params["homophily_state"] = 0
    #NO carbon_price
    params["carbon_price_increased_lower"] = 0    
    params_list_no_tau_no_homophilly = produce_param_list_stochastic(params, property_values_list, property_varied)

    #Low carbon price
    params["carbon_price_increased_lower"] = 0.1   
    params_list_low_tau_no_homophilly = produce_param_list_stochastic(params, property_values_list, property_varied)

    #High carbon rpice
    params["carbon_price_increased_lower"] = 0.5   
    params_list_high_tau_no_homophilly = produce_param_list_stochastic(params, property_values_list, property_varied)

    params_list_no_homophilly = params_list_no_tau_no_homophilly + params_list_low_tau_no_homophilly + params_list_high_tau_no_homophilly


    #NO carbon_price
    params["homophily_state"] = 1
    params["carbon_price_increased_lower"] = 0    
    params_list_no_tau_homophilly = produce_param_list_stochastic(params, property_values_list, property_varied)

    #Low carbon price
    params["carbon_price_increased_lower"] = 0.1   
    params_list_low_tau_homophilly = produce_param_list_stochastic(params, property_values_list, property_varied)

    #High carbon rpice
    params["carbon_price_increased_lower"] = 0.5   
    params_list_high_tau_homophilly = produce_param_list_stochastic(params, property_values_list, property_varied)

    params_list_homophilly = params_list_no_tau_homophilly + params_list_low_tau_homophilly + params_list_high_tau_homophilly
    
    print("TOTAL RUNS", len(params_list_no_homophilly) + len(params_list_homophilly))
    
    emissions_stock_serial_no_homophilly = multi_emissions_stock(params_list_no_homophilly)
    emissions_array_no_homophilly = emissions_stock_serial_no_homophilly.reshape(3, property_reps, params["seed_reps"])#3 is for the 3 differents states

    emissions_stock_serial_homophilly = multi_emissions_stock(params_list_homophilly)
    emissions_array_homophilly = emissions_stock_serial_homophilly.reshape(3, property_reps, params["seed_reps"])#3 is for the 3 differents states

    save_object(emissions_array_homophilly, fileName + "/Data", "emissions_array_homophilly")
    save_object(emissions_array_no_homophilly, fileName + "/Data", "emissions_array_no_homophilly")
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_SBM_density.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_SBM_density.json",
        RUN_TYPE = 5
)

