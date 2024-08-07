# imports
from package.resources.run import  identity_timeseries_run
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)

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

def main(
    base_params
) -> str: 

    root = "homo_four"
    fileName = produce_name_datetime(root)
    #pyperclip.copy(fileName)
    print("fileName:", fileName)

    params_list = []
    #case1
    base_params["homophily_state"] = 0
    base_params["coherance_state"] = 0
    #params_list.append(deepcopy(base_params))
    params_seeds = produce_param_list_only_stochastic_homo(base_params)
    params_list.extend(params_seeds)
    #case2
    base_params["homophily_state"] = 1
    base_params["coherance_state"] = 0
    #params_list.append(deepcopy(base_params))
    params_seeds = produce_param_list_only_stochastic_homo(base_params)
    params_list.extend(params_seeds)
    #case3
    base_params["homophily_state"] = 0
    base_params["coherance_state"] = 1
    #params_list.append(deepcopy(base_params))
    params_seeds = produce_param_list_only_stochastic_homo(base_params)
    params_list.extend(params_seeds)
    #case4
    base_params["homophily_state"] = 1
    base_params["coherance_state"] = 1
    #params_list.append(deepcopy(base_params))
    params_seeds = produce_param_list_only_stochastic_homo(base_params)
    params_list.extend(params_seeds)

    print("Total runs:",len(params_list))
    Data_array_flat = identity_timeseries_run(params_list)
    time_steps = Data_array_flat[0].shape[0]

    Data_array = Data_array_flat.reshape(4, base_params["seed_reps"],time_steps, base_params["N"] )

    createFolder(fileName)
    save_object(Data_array, fileName + "/Data", "Data_array")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':
    
    base_params = {
    "preferences_seed": 72,
    "network_structure_seed": 89, 
    "phi_lower": 0.03,
    "network_type": "SW",
    "carbon_price_increased_lower": 0,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "heterogenous_intrasector_preferences_state": 1,
    "heterogenous_carbon_price_state": 0,
    "heterogenous_sector_substitutabilities_state": 0,
    "SBM_block_heterogenous_individuals_substitutabilities_state": 0,
    "heterogenous_phi_state": 0,
    "imperfect_learning_state": 1,
    "imitation_state": "consumption",
    "vary_seed_state": "multi",
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
    "low_carbon_substitutability_lower": 2, 
    "low_carbon_substitutability_upper": 2, 
    "a_preferences": 2, 
    "b_preferences": 2, 
    "clipping_epsilon_init_preference": 1e-5, 
    "confirmation_bias": 5, 
    "init_carbon_price": 0, 
    "BA_nodes": 160,
    #"BA_density":0.1,
    "BA_green_or_brown_hegemony": 0,
    "SBM_block_num": 2,
    "SBM_network_density_input_intra_block": 0.2,
    "SBM_network_density_input_inter_block": 0.005,
    "SW_network_density": 0.1,
    "SW_prob_rewire": 0.1
    }

    fileName = main(base_params=base_params)
