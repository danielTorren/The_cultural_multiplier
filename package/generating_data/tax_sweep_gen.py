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
from package.resources.run import emissions_parallel_run, emissions_parallel_run_timeseries
from copy import deepcopy

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(np.log10(variable_parameters_dict["property_min"]),np.log10( variable_parameters_dict["property_max"]), variable_parameters_dict["property_reps"])
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

def arrange_scenarios_no_tax(base_params,scenarios):
    base_params_copy = deepcopy(base_params)
    base_params_copy["carbon_price_increased_lower"] = 0# no carbon tax
    base_params_copy["ratio_preference_or_consumption_state"] = 0 #WE ASSUME Consumption BASE LEARNING

    params_list = []
    #scenarios = [
    # "fixed_preferences",
    # "uniform_network_weighting", 
    # "static_socially_determined_weights",
    # "static_culturally_determined_weights", 
    # "dynamic_socially_determined_weights", 
    # "dynamic_culturally_determined_weights" 
    # ]
    ###### WITHOUT CARBON TAX
    # 1. Run with fixed preferences, Emissions: [S_n]
    if "fixed_preferences" in scenarios:
        base_params_copy_1 = deepcopy(base_params_copy)
        base_params_copy_1["alpha_change_state"] = "fixed_preferences"
        params_sub_list_1 = produce_param_list_just_stochastic(base_params_copy_1)
        params_list.extend(params_sub_list_1)

    # 2. Run with fixed network weighting uniform, Emissions: [S_n]
    if "uniform_network_weighting" in scenarios:
        base_params_copy_2 = deepcopy(base_params_copy)
        base_params_copy_2["alpha_change_state"] = "uniform_network_weighting"
        base_params_copy_2["confirmation_bias"] = 0
        params_sub_list_2 = produce_param_list_just_stochastic(base_params_copy_2)
        params_list.extend(params_sub_list_2)

    # 3. Run with fixed network weighting socially determined, Emissions: [S_n]
    if "static_socially_determined_weights" in scenarios:
        base_params_copy_3 = deepcopy(base_params_copy)
        base_params_copy_3["alpha_change_state"] = "static_socially_determined_weights"
        params_sub_list_3 = produce_param_list_just_stochastic(base_params_copy_3)
        params_list.extend(params_sub_list_3)

    # 4. Run with fixed network weighting culturally determined, Emissions: [S_n]
    if "static_culturally_determined_weights" in scenarios:
        base_params_copy_4 = deepcopy(base_params_copy)
        base_params_copy_4["alpha_change_state"] = "static_culturally_determined_weights"
        params_sub_list_4 = produce_param_list_just_stochastic(base_params_copy_4)
        params_list.extend(params_sub_list_4)

    # 5. Run with social learning, Emissions: [S_n]
    if "dynamic_socially_determined_weights" in scenarios:
        base_params_copy_5 = deepcopy(base_params_copy)
        base_params_copy_5["alpha_change_state"] = "dynamic_socially_determined_weights"
        params_sub_list_5 = produce_param_list_just_stochastic(base_params_copy_5)
        params_list.extend(params_sub_list_5)

    # 6.  Run with cultural learning, Emissions: [S_n]
    if "dynamic_culturally_determined_weights" in scenarios:
        base_params_copy_6 = deepcopy(base_params_copy)
        base_params_copy_6["alpha_change_state"] = "dynamic_culturally_determined_weights"
        params_sub_list_6 = produce_param_list_just_stochastic(base_params_copy_6)
        params_list.extend(params_sub_list_6)

    if "fully_connected_uniform" in scenarios:
        base_params_copy_7 = deepcopy(base_params_copy)
        base_params_copy_7["alpha_change_state"] = "fixed_preferences"
        base_params_copy_7["network_density"] = 1
        params_sub_list_7 = produce_param_list_just_stochastic(base_params_copy_7)
        params_list.extend(params_sub_list_7)

    return params_list

def produce_param_list_scenarios_tax(params: dict, property_list: list, property: str) -> list[dict]:
    params_list = []
    for i in property_list:
        params[property] = i
        for v in range(params["seed_reps"]):
            params["set_seed"] = int(v+1)
            params_list.append(params.copy())  
    return params_list

def arrange_scenarios_tax(base_params_tax, carbon_tax_vals,scenarios):
    base_params_tax_copy = deepcopy(base_params_tax)

    base_params_tax_copy["ratio_preference_or_consumption_state"] = 0 #WE ASSUME Consumption BASE LEARNING

    params_list = []

    ###### WITHOUT CARBON TAX

    # 1. Run with fixed preferences, Emissions: [S_n]
    if "fixed_preferences" in scenarios:
        base_params_copy_1 = deepcopy(base_params_tax_copy)
        base_params_copy_1["alpha_change_state"] = "fixed_preferences"
        params_sub_list_1 = produce_param_list_scenarios_tax(base_params_copy_1, carbon_tax_vals,"carbon_price_increased_lower")
        params_list.extend(params_sub_list_1)

    # 2. Run with fixed network weighting uniform, Emissions: [S_n]
    if "uniform_network_weighting" in scenarios:
        base_params_copy_2 = deepcopy(base_params_tax_copy)
        base_params_copy_2["alpha_change_state"] = "uniform_network_weighting"
        base_params_copy_2["confirmation_bias"] = 0
        params_sub_list_2 = produce_param_list_scenarios_tax(base_params_copy_2, carbon_tax_vals,"carbon_price_increased_lower")
        params_list.extend(params_sub_list_2)

    # 3. Run with fixed network weighting socially determined, Emissions: [S_n]
    if "static_socially_determined_weights" in scenarios:
        base_params_copy_3 = deepcopy(base_params_tax_copy)
        base_params_copy_3["alpha_change_state"] = "static_socially_determined_weights"
        params_sub_list_3 = produce_param_list_scenarios_tax(base_params_copy_3, carbon_tax_vals,"carbon_price_increased_lower")
        params_list.extend(params_sub_list_3)

    # 4. Run with fixed network weighting culturally determined, Emissions: [S_n]
    if "static_culturally_determined_weights" in scenarios:
        base_params_copy_4 = deepcopy(base_params_tax_copy)
        base_params_copy_4["alpha_change_state"] = "static_culturally_determined_weights"
        params_sub_list_4 = produce_param_list_scenarios_tax(base_params_copy_4, carbon_tax_vals,"carbon_price_increased_lower")
        params_list.extend(params_sub_list_4)

    # 5. Run with social learning, Emissions: [S_n]
    if "dynamic_socially_determined_weights" in scenarios:
        base_params_copy_5 = deepcopy(base_params_tax_copy)
        base_params_copy_5["alpha_change_state"] = "dynamic_socially_determined_weights"
        params_sub_list_5 = produce_param_list_scenarios_tax(base_params_copy_5, carbon_tax_vals,"carbon_price_increased_lower")
        params_list.extend(params_sub_list_5)

    # 6.  Run with cultural learning, Emissions: [S_n]
    if "dynamic_culturally_determined_weights" in scenarios:
        base_params_copy_6 = deepcopy(base_params_tax_copy)
        base_params_copy_6["alpha_change_state"] = "dynamic_culturally_determined_weights"
        params_sub_list_6 = produce_param_list_scenarios_tax(base_params_copy_6, carbon_tax_vals,"carbon_price_increased_lower")
        params_list.extend(params_sub_list_6)

    if "fully_connected_uniform" in scenarios:
        base_params_copy_7 = deepcopy(base_params_tax_copy)
        base_params_copy_7["network_density"] = 1
        base_params_copy_7["alpha_change_state"] = "uniform_network_weighting"
        params_sub_list_7 = produce_param_list_scenarios_tax(base_params_copy_7, carbon_tax_vals,"carbon_price_increased_lower")
        params_list.extend(params_sub_list_7)

    return params_list

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_tau_vary.json",
        print_simu = 1,
        scenarios = ["fixed_preferences","uniform_network_weighting", "static_socially_determined_weights","static_culturally_determined_weights", "dynamic_socially_determined_weights", "dynamic_culturally_determined_weights" ],
        RUN_TYPE = 1
        ) -> str: 

    scenario_reps = len(scenarios)

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
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

    root = "tax_sweep"
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
    # 3. Run with fixed network weighting socially determined, Emissions: [S_n]
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
    params_list_no_tax = arrange_scenarios_no_tax(params,scenarios)
    params_list_tax = arrange_scenarios_tax(params,property_values_list,scenarios)

    #quit()
    print("Total runs: ",len(params_list_tax) + len(params_list_no_tax))
    
    #RESULTS
    if RUN_TYPE == 1:
        emissions_stock_no_tax_flat = emissions_parallel_run(params_list_no_tax)
        print("NO TAX DONE")
        emissions_stock_tax_flat = emissions_parallel_run(params_list_tax)

        #unpack_results into scenarios and seeds
        emissions_stock_no_tax = emissions_stock_no_tax_flat.reshape(scenario_reps,params["seed_reps"])
        emissions_stock_tax = emissions_stock_tax_flat.reshape(scenario_reps,property_reps,params["seed_reps"])
    elif RUN_TYPE == 2:
        emissions_stock_no_tax_flat = emissions_parallel_run_timeseries(params_list_no_tax)
        print("NO TAX DONE")
        emissions_stock_tax_flat = emissions_parallel_run_timeseries(params_list_tax)

        #unpack_results into scenarios and seeds
        emissions_stock_no_tax = emissions_stock_no_tax_flat.reshape(scenario_reps,params["seed_reps"], params["carbon_price_duration"])
        emissions_stock_tax = emissions_stock_tax_flat.reshape(scenario_reps,property_reps,params["seed_reps"], params["carbon_price_duration"])

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    #quit()
    ##################################
    #save data

    createFolder(fileName)

    save_object(emissions_stock_no_tax, fileName + "/Data", "emissions_stock_no_tax")
    save_object(emissions_stock_tax, fileName + "/Data", "emissions_stock_tax")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")
    save_object(scenarios, fileName + "/Data", "scenarios")
    save_object(RUN_TYPE,  fileName + "/Data", "RUN_TYPE")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",#"package/constants/base_params_tau_vary_timeseries.json",#"package/constants/base_params_tau_vary.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_tau_vary.json",#"package/constants/oneD_dict_tau_vary_timeseries.json", #"package/constants/oneD_dict_tau_vary.json",
        #scenarios = ["uniform_network_weighting", "dynamic_socially_determined_weights"],
        scenarios = ["fully_connected_uniform","fixed_preferences","uniform_network_weighting","static_culturally_determined_weights", "dynamic_culturally_determined_weights" ],
        RUN_TYPE = 1
    )
    #["fixed_preferences","uniform_network_weighting", "static_socially_determined_weights","static_culturally_determined_weights", "dynamic_socially_determined_weights", "dynamic_culturally_determined_weights" ],
    #        scenarios = ["fixed_preferences","uniform_network_weighting", "static_socially_determined_weights","static_culturally_determined_weights", "dynamic_socially_determined_weights", "dynamic_culturally_determined_weights" ],

