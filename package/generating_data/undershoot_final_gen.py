# imports
from package.resources.run import  emissions_parallel_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    produce_param_list_stochastic_multi
)
import numpy as np
from copy import deepcopy

def arrange_scenarios(base_params_tax, param_vals,scenarios, param_varied):
    base_params_tax_copy = deepcopy(base_params_tax)
    params_list = []

    # 1. Run with fixed preferences, Emissions: [S_n]
    if "fixed_preferences" in scenarios:
        base_params_copy_1 = deepcopy(base_params_tax_copy)
        base_params_copy_1["alpha_change_state"] = "fixed_preferences"
        params_sub_list_1 = produce_param_list_stochastic_multi(base_params_copy_1, param_vals,param_varied)
        params_list.extend(params_sub_list_1)

    # 5. Run with social learning, Emissions: [S_n]
    if "dynamic_socially_determined_weights" in scenarios:
        base_params_copy_5 = deepcopy(base_params_tax_copy)
        base_params_copy_5["alpha_change_state"] = "dynamic_socially_determined_weights"
        params_sub_list_5 = produce_param_list_stochastic_multi(base_params_copy_5, param_vals,param_varied)
        params_list.extend(params_sub_list_5)

    # 6.  Run with cultural learning, Emissions: [S_n]
    if "dynamic_identity_determined_weights" in scenarios:
        base_params_copy_6 = deepcopy(base_params_tax_copy)
        base_params_copy_6["alpha_change_state"] = "dynamic_identity_determined_weights"
        params_sub_list_6 = produce_param_list_stochastic_multi(base_params_copy_6, param_vals,param_varied)
        params_list.extend(params_sub_list_6)

    return params_list

def produce_param_list_only_stochastic_homo(params: dict) -> list[dict]:
    seeds_labels = ["shuffle_homophily_seed", "shuffle_coherance_seed"]
    params_list = []
    for j in range(params["seed_reps"]):
        for k, label in enumerate(seeds_labels):
            params[label] = int(10*k + j + 1)
        params_list.append(
            params.copy()
        )  
    return params_list

def generate_params_list(base_params, produce_param_list_only_stochastic_homo, homophily_values, coherance_values):
    params_list = []
    
    for homophily_state in homophily_values:
        for coherance_state in coherance_values:
            base_params["homophily_state"] = homophily_state
            base_params["coherance_state"] = coherance_state
            params_seeds = produce_param_list_only_stochastic_homo(base_params)
            params_list.extend(params_seeds)
    
    return params_list

def main(
    base_params,
    scenarios
) -> str: 

    root = "undershoot"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    reps = 10
    param_vals = np.linspace(0, 5, reps)#np.logspace(np.log10(0.001), np.log10(0.05), reps)# np.linspace(0.1, 4, reps)
    param_varied = "confirmation_bias"#"phi_lower"#"a_preferences"

    params_list = arrange_scenarios(base_params, param_vals, scenarios, param_varied)
    print("Total runs:",len(params_list))

    #Data_array_flat = identity_timeseries_run(params_list)
    
    Data_flat= emissions_parallel_run(params_list)

    print("RUNS DONE")

    Data_array =  Data_flat.reshape(len(scenarios), reps, base_params["seed_reps"])

    createFolder(fileName)

    save_object(base_params, fileName + "/Data", "base_params")
    save_object(Data_array, fileName + "/Data", "Data_array")
    save_object(scenarios, fileName + "/Data", "scenarios")
    save_object(param_vals, fileName + "/Data", "param_vals")
    save_object(param_varied, fileName + "/Data", "param_varied" )

    return fileName

if __name__ == '__main__':
    
    base_params = {
    "network_type": "SW",
    "coherance_state": 0,
    "homophily_state": 0,
    "phi_lower": 0.03,
    "carbon_price_increased_lower": 0,
    "save_timeseries_data_state": 0,
    "compression_factor_state": 1,

   
    
    
    
    "imitation_state": "consumption",
    "vary_seed_state": "multi",
    "seed_reps": 5,#100,
    "carbon_price_duration": 1,#360, 
    "burn_in_duration": 0, 
    "N": 500,#3000, 
    "M": 2, 
    "sector_substitutability": 2, 
    "low_carbon_substitutability_lower": 2, 
    "low_carbon_substitutability_upper": 2, 
    "a_preferences": 2, 
    "b_preferences": 2, 
    "clipping_epsilon_init_preference": 1e-5, 
    #"confirmation_bias": 5, 
    "init_carbon_price": 0, 
    "BA_density":0.1,
    #"BA_density":0.1,
    "BA_green_or_brown_hegemony": 0,
    "SBM_block_num": 2,
    "SBM_network_density_input_intra_block": 0.2,
    "SBM_network_density_input_inter_block": 0.005,
    "SW_network_density": 0.1,
    "SW_prob_rewire": 0.1
    }

    scenarios = ["fixed_preferences","dynamic_socially_determined_weights", "dynamic_identity_determined_weights" ]

    fileName = main(base_params=base_params, scenarios= scenarios)
