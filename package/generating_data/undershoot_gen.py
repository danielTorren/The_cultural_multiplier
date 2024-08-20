# imports
from package.resources.run import  identity_timeseries_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime, 
    produce_param_list_only_stochastic_named
)
import numpy as np
from copy import deepcopy

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

def arrange_scenarios_tax(base_params_tax, scenarios):
    
    seeds_labels = ["preferences_seed", "network_structure_seed", "shuffle_homophily_seed", "shuffle_coherance_seed"]

    base_params_tax_copy = deepcopy(base_params_tax)
    params_list = []

    # 1. Run with fixed preferences, Emissions: [S_n]
    if "fixed_preferences" in scenarios:
        base_params_copy_1 = deepcopy(base_params_tax_copy)
        base_params_copy_1["alpha_change_state"] = "fixed_preferences"
        params_sub_list_1 = produce_param_list_only_stochastic_named(base_params_copy_1,seeds_labels)
        params_list.extend(params_sub_list_1)

    # 5. Run with social learning, Emissions: [S_n]
    if "dynamic_socially_determined_weights" in scenarios:
        base_params_copy_5 = deepcopy(base_params_tax_copy)
        base_params_copy_5["alpha_change_state"] = "dynamic_socially_determined_weights"
        params_sub_list_5 = produce_param_list_only_stochastic_named(base_params_copy_5,seeds_labels)
        params_list.extend(params_sub_list_5)

    # 6.  Run with cultural learning, Emissions: [S_n]
    if "dynamic_identity_determined_weights" in scenarios:
        base_params_copy_6 = deepcopy(base_params_tax_copy)
        base_params_copy_6["alpha_change_state"] = "dynamic_identity_determined_weights"
        params_sub_list_6 = produce_param_list_only_stochastic_named(base_params_copy_6,seeds_labels)
        params_list.extend(params_sub_list_6)
    
    return params_list

def main(
    base_params
) -> str: 

    root = "undershoot"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    scenarios = ["fixed_preferences","dynamic_socially_determined_weights", "dynamic_identity_determined_weights" ]

    base_params["network_type"] = "SW"
    params_list = arrange_scenarios_tax(base_params,scenarios)
    print("Total runs:",len(params_list))

    Data_array_flat = identity_timeseries_run(params_list)

    print("RUNS DONE")
    
    time_steps = Data_array_flat[0].shape[0]

    Data_array = Data_array_flat.reshape(len(scenarios), base_params["seed_reps"], time_steps, base_params["N"] )

    seeds, time_steps, N = Data_array.shape[1], Data_array.shape[2], Data_array.shape[3]
    history_time = np.arange(time_steps)  # Assuming the same time steps for all data
    time_tile = np.tile(history_time, N * seeds)

    bin_num = 100

    h_list = []

    for i in range(len(scenarios)):
        data_subfigure = Data_array[i]
        data_trans = data_subfigure.transpose(0, 2, 1)
        combined_data = data_trans.reshape(seeds * N, time_steps)
        data_flat = combined_data.flatten()

        h = np.histogram2d(time_tile, data_flat, bins=[time_steps, bin_num], density=True)
    
        h_list.append(h)

    createFolder(fileName)

    #save_object(Data_array, fileName + "/Data", "Data_array")

    save_object(base_params, fileName + "/Data", "base_params")
    save_object(h_list, fileName + "/Data", "h_list")

    return fileName

if __name__ == '__main__':
    
    base_params = {
    "coherance_state": 0.9,
    "homophily_state": 0,
    "phi_lower": 0.03,
    "carbon_price_increased_lower": 0,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "heterogenous_intrasector_preferences_state": 1,
    "heterogenous_carbon_price_state": 0,
    "heterogenous_sector_substitutabilities_state": 0,
    "SBM_block_heterogenous_individuals_substitutabilities_state": 0,
    "heterogenous_phi_state": 0,
    "imitation_state": "consumption",
    "vary_seed_state": "multi",
    "seed_reps": 100,
    "carbon_price_duration": 360, 
    "burn_in_duration": 0, 
    "N": 3000,#3000, 
    "M": 2, 
    "sector_substitutability": 2, 
    "low_carbon_substitutability_lower": 2, 
    "low_carbon_substitutability_upper": 2, 
    "a_preferences": 2, 
    "b_preferences": 2, 
    "clipping_epsilon_init_preference": 1e-5, 
    "confirmation_bias": 5, 
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

    fileName = main(base_params=base_params)
