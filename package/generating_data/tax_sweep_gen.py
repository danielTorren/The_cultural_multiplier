"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
import time
from copy import deepcopy
import json
import numpy as np
from package.resources.utility import createFolder, produce_name_datetime, save_object
from package.resources.run import emissions_parallel_run

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_values_list 

def produce_param_list_just_stochastic(params: dict) -> list[dict]:
    params_list = []
    for v in range(params["seed_reps"]):
        params["set_seed"] = int(v+1)
        params_list.append(
            params.copy()
        )  
    return params_list

def produce_param_list_scenarios_no_tax(base_params):

    base_params["carbon_price_increased"] = 0# no carbon tax
    base_params["ratio_preference_or_consumption"] = 0 #WE ASSUME Consumption BASE LEARNING

    params_list = []

    ###### WITHOUT CARBON TAX
    # 1. Run with fixed preferences, Emissions: [S_n]
    base_params_copy_1 = deepcopy(base_params)
    base_params_copy_1["alpha_change"] = "static_preferences"
    params_sub_list_1 = produce_param_list_just_stochastic(base_params_copy_1)
    params_list.extend(params_sub_list_1)

    # 2. Run with fixed network weighting uniform, Emissions: [S_n]
    base_params_copy_2 = deepcopy(base_params)
    base_params_copy_2["alpha_change"] = "static_culturally_determined_weights"
    base_params_copy_2["confirmation_bias"] = 0
    params_sub_list_2 = produce_param_list_just_stochastic(base_params_copy_2)
    params_list.extend(params_sub_list_2)

    # 3. Run with fixed network weighting socially determined, Emissions: [S_n]
    #base_params_copy_3 = deepcopy(base_params)
    #base_params_copy_3["alpha_change"] = "static_socially_determined_weights"
    #params_sub_list_3 = produce_param_list_just_stochastic(base_params_copy_3)
    #params_list.extend(params_sub_list_3)

    # 4. Run with fixed network weighting culturally determined, Emissions: [S_n]
    base_params_copy_4 = deepcopy(base_params)
    base_params_copy_4["alpha_change"] = "static_culturally_determined_weights"
    params_sub_list_4 = produce_param_list_just_stochastic(base_params_copy_4)
    params_list.extend(params_sub_list_4)

    # 5. Run with social learning, Emissions: [S_n]
    base_params_copy_5 = deepcopy(base_params)
    base_params_copy_5["alpha_change"] = "dynamic_socially_determined_weights"
    params_sub_list_5 = produce_param_list_just_stochastic(base_params_copy_5)
    params_list.extend(params_sub_list_5)

    # 6.  Run with cultural learning, Emissions: [S_n]
    base_params_copy_6 = deepcopy(base_params)
    base_params_copy_6["alpha_change"] = "dynamic_culturally_determined_weights"
    params_sub_list_6 = produce_param_list_just_stochastic(base_params_copy_5)
    params_list.extend(params_sub_list_6)

    scenarios_reps = 5
    return params_list, scenarios_reps

def produce_param_list_scenarios_tax(params: dict, property_list: list, property: str) -> list[dict]:
    params_list = []
    for i in property_list:
        params[property] = i
        for v in range(params["seed_reps"]):
            params["set_seed"] = int(v+1)
            params_list.append(params.copy())  
    return params_list

def produce_param_list_scenarios_carbon_tax(base_params, carbon_tax_vals):

    base_params["ratio_preference_or_consumption"] = 0 #WE ASSUME Consumption BASE LEARNING

    params_list = []

    ###### WITHOUT CARBON TAX
    # 1. Run with fixed preferences, Emissions: [S_n]
    base_params_copy_1 = deepcopy(base_params)
    base_params_copy_1["alpha_change"] = "static_preferences"
    params_sub_list_1 = produce_param_list_scenarios_tax(base_params_copy_1, carbon_tax_vals,"carbon_price_increased")
    params_list.extend(params_sub_list_1)

    # 2. Run with fixed network weighting uniform, Emissions: [S_n]
    base_params_copy_2 = deepcopy(base_params)
    base_params_copy_2["alpha_change"] = "static_culturally_determined_weights"
    base_params_copy_2["confirmation_bias"] = 0
    params_sub_list_2 = produce_param_list_scenarios_tax(base_params_copy_2, carbon_tax_vals,"carbon_price_increased")
    params_list.extend(params_sub_list_2)

    # 3. Run with fixed network weighting socially determined, Emissions: [S_n]
    #base_params_copy_3 = deepcopy(base_params)
    #base_params_copy_3["alpha_change"] = "static_socially_determined_weights"
    #params_sub_list_3 = produce_param_list_stochastic_tax(base_params_copy_3, carbon_tax_vals,"carbon_price_increased")
    #params_list.extend(params_sub_list_3)

    # 4. Run with fixed network weighting culturally determined, Emissions: [S_n]
    base_params_copy_4 = deepcopy(base_params)
    base_params_copy_4["alpha_change"] = "static_culturally_determined_weights"
    params_sub_list_4 = produce_param_list_scenarios_tax(base_params_copy_4, carbon_tax_vals,"carbon_price_increased")
    params_list.extend(params_sub_list_4)

    # 5. Run with social learning, Emissions: [S_n]
    base_params_copy_5 = deepcopy(base_params)
    base_params_copy_5["alpha_change"] = "dynamic_socially_determined_weights"
    params_sub_list_5 = produce_param_list_scenarios_tax(base_params_copy_5, carbon_tax_vals,"carbon_price_increased")
    params_list.extend(params_sub_list_5)

    # 6.  Run with cultural learning, Emissions: [S_n]
    base_params_copy_6 = deepcopy(base_params)
    base_params_copy_6["alpha_change"] = "dynamic_culturally_determined_weights"
    params_sub_list_6 = produce_param_list_scenarios_tax(base_params_copy_5, carbon_tax_vals,"carbon_price_increased")
    params_list.extend(params_sub_list_6)

        scenarios_reps = 5
    return params_list, scenarios_reps

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
        print_simu = 1,
        ) -> str: 

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption",
    property_min = var_params["property_min"]#0,
    property_max = var_params["property_max"]#1,
    property_reps = var_params["property_reps"]#10,
    property_varied_title = var_params["property_varied_title"]# #"A to Omega ratio"

    property_values_list = generate_vals(
        var_params
    )
    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "emission_target_sweep"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    
    ##################################
    #Hyper_
    #No burn in period of model. Will be faster
    #Stochastic runs number = S_n
    #Carbon tax values = tau_n
    #
    #Structure of the experiment:
    ###### WITHOUT CARBON TAX
    # 1. Run with fixed preferences, Emissions: [S_n]
    # 2. Run with fixed network weighting uniform, Emissions: [S_n]
    #DONT HAVE A WAY TO DO THIS ATM ######################### 3. Run with fixed network weighting socially determined, Emissions: [S_n]
    # 4. Run with fixed network weighting culturally determined, Emissions: [S_n]
    # 5. Run with social learning, Emissions: [S_n]
    # 6.  Run with cultural learning, Emissions: [S_n]

    #WITH CARBON TAX
    # 7. Run with fixed preferences, Emissions: [S_n]
    # 8. Run with fixed network weighting uniform, Emissions: [S_n]
    # 9. Run with fixed network weighting socially determined, Emissions: [S_n]
    # 10. Run with fixed network weighting culturally determined, Emissions: [S_n]
    # 11. Run with social learning, Emissions: [S_n]
    # 12.  Run with cultural learning, Emissions: [S_n]

    #12 runs total * the number of seeds (Unsure if 2,3,4 and 8,9,10 are necessary but they do isolate the dyanmic model aspects)

    #Gen params lists
    params_list_no_tax, scenario_reps_no_tax = produce_param_list_scenarios_no_tax(params)
    params_list_tax, scenario_reps_no_tax = produce_param_list_scenarios_tax(params)

    #RESULTS
    emissions_no_tax_flat = emissions_parallel_run(params_list_no_tax)
    emissions_tax_flat = emissions_parallel_run(params_list_tax)

    #unpack_results into scenarios and seeds
    emissions_no_tax = emissions_no_tax_flat.reshape(scenario_reps,params["seed_reps"])





    ##################################
    #save data

    createFolder(fileName)

    save_object(emissions_stock_seeds, fileName + "/Data", "emissions_stock_seeds")
    save_object(tau_seeds_no_preference_change, fileName + "/Data", "tau_seeds_no_preference_change")
    save_object(tau_seeds_preference_change_matrix, fileName + "/Data", "tau_seeds_preference_change_matrix")
    save_object(multiplier_matrix_attitude_preference_change, fileName + "/Data", "multiplier_matrix_attitude_preference_change") 
    save_object(multiplier_matrix_no_preference_change, fileName + "/Data", "multiplier_matrix_no_preference_change") 
    save_object(params, fileName + "/Data", "base_params")
    save_object(reduction_prop, fileName + "/Data", "reduction_prop")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")


    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_mu_target.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_mu_target.json",
        reduction_prop = 0.5,
        carbon_price_duration = 1000,
        static_weighting_run = 1
)

