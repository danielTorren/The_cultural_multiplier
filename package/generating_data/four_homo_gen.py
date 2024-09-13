# imports
from package.resources.run import  identity_timeseries_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)
import numpy as np

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
    base_params
) -> str: 

    root = "homo_four"
    fileName = produce_name_datetime(root)
    #pyperclip.copy(fileName)
    print("fileName:", fileName)

    homophily_values = [0, 0.9, 1]
    coherance_values = [0, 0.9, 1]

    total_reps = len(homophily_values)*len(coherance_values)

    params_list = generate_params_list(base_params, produce_param_list_only_stochastic_homo, homophily_values, coherance_values)

    print("Total runs:",len(params_list))
    Data_array_flat = identity_timeseries_run(params_list)
    time_steps = Data_array_flat[0].shape[0]

    Data_array = Data_array_flat.reshape(total_reps, base_params["seed_reps"],time_steps, base_params["N"] )

    seeds, time_steps, N = Data_array.shape[1], Data_array.shape[2], Data_array.shape[3]
    history_time = np.arange(time_steps)  # Assuming the same time steps for all data
    time_tile = np.tile(history_time, N * seeds)

    bin_num = 100

    h_list = []

    for i in range(total_reps):
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
    save_object(homophily_values, fileName + "/Data", "homophily_values")
    save_object(coherance_values, fileName + "/Data", "coherance_values")

    return fileName

if __name__ == '__main__':
    
    base_params = {
    "preferences_seed": 72,
    "network_structure_seed": 89, 
    "phi": 0.02,
    "network_type": "SW",
    "carbon_price_increased": 0,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "imitation_state": "consumption",
    "alpha_change_state": "dynamic_identity_determined_weights",
    "seed_reps": 100,
    "network_structure_seed": 1, 
    "preferences_seed": 10, 
    "shuffle_homophily_seed": 20,
    "shuffle_coherance_seed": 30,
    "carbon_price_duration": 360, 
    "burn_in_duration": 0, 
    "N": 3000, 
    "M": 2, 
    "sector_substitutability": 2, 
    "low_carbon_substitutability": 4, 
     
    "a_preferences": 2, 
    "b_preferences": 2, 
    "clipping_epsilon_init_preference": 1e-5, 
    "confirmation_bias": 5, 
    "init_carbon_price": 0, 
    "SF_density":0.1,
    "SF_green_or_brown_hegemony": 0,
    "SBM_block_num": 2,
    "SBM_network_density_input_intra_block": 0.2,
    "SBM_network_density_input_inter_block": 0.005,
    "SW_network_density": 0.1,
    "SW_prob_rewire": 0.1
    }

    fileName = main(base_params=base_params)
