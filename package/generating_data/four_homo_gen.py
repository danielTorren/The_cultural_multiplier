"""Runs a single simulation to produce data which is saved

A module that use dictionary of data for the simulation run. The single shot simulztion is run
for a given initial set seed.



Created: 10/10/2022
"""
# imports
from package.resources.run import generate_data
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)
from package.plotting_data import four_homo_plot
import pyperclip

def main(
    base_params
) -> str: 

    root = "homo_four"
    fileName = produce_name_datetime(root)
    pyperclip.copy(fileName)
    print("fileName:", fileName)

    #case1
    base_params["homophily_state"] = 0
    base_params["coherance_state"] = 0
    Data1 = generate_data(base_params)  # run the simulation
    #case2
    base_params["homophily_state"] = 1
    base_params["coherance_state"] = 0
    Data2 = generate_data(base_params)  # run the simulation
    #case3
    base_params["homophily_state"] = 0
    base_params["coherance_state"] = 1
    Data3 = generate_data(base_params)  # run the simulation
    #case4
    base_params["homophily_state"] = 1
    base_params["coherance_state"] = 1
    Data4 = generate_data(base_params)  # run the simulation

    Data_list = [Data1, Data2, Data3, Data4]
    createFolder(fileName)
    save_object(Data_list, fileName + "/Data", "Data_list")
    
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':
    
    base_params = {
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
    "seed_reps": 25,
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

    RUN_PLOT = 1

    if RUN_PLOT:
        four_homo_plot.main(fileName = fileName)
