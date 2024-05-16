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
from package.plotting_data import single_experiment_plot
import pyperclip

def main(
    base_params
) -> str: 

    root = "single_experiment"
    fileName = produce_name_datetime(root)
    pyperclip.copy(fileName)
    print("fileName:", fileName)

    Data = generate_data(base_params)  # run the simulation
    #print(Data.average_identity)

    createFolder(fileName)
    save_object(Data, fileName + "/Data", "social_network")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':
    
    #base_params = {'save_timeseries_data_state': 1, 'compression_factor_state': 1, 'heterogenous_intrasector_preferences_state': 1, 'heterogenous_carbon_price_state': 1, 'heterogenous_sector_substitutabilities_state': 1, 'SBM_block_heterogenous_individuals_substitutabilities_state': 0, 'heterogenous_phi_state': 0, 'imperfect_learning_state': 1, 'imitation_state': 'consumption', 'alpha_change_state': 'dynamic_identity_determined_weights', 'vary_seed_state': 'network', 'seed_reps': 25, 'network_structure_seed': 8, 'preferences_seed': 14, 'shuffle_homophily_seed': 20, 'learning_seed': 10, 'carbon_price_duration': 1000, 'burn_in_duration': 0, 'M': 2, 'clipping_epsilon_init_preference': 1e-05, 'init_carbon_price': 0, 'BA_nodes': 11, 'BA_green_or_brown_hegemony': 0, 'SBM_block_num': 2, 'SBM_network_density_input_intra_block': 0.2, 'SBM_network_density_input_inter_block': 0.005, 'SW_network_density': 0.1, 'SW_prob_rewire': 0.1, 'network_type': 'SW', 'phi_lower': 0.79818115234375, 'carbon_price_increased_lower': 1.568603515625, 'carbon_price_increased_upper': 3.670654296875, 'N': 399.0, 'sector_substitutability': 2.3463623046875, 'low_carbon_substitutability_lower': 3.9688232421875, 'low_carbon_substitutability_upper': 4.7134033203125, 'confirmation_bias': 21.13037109375, 'homophily_state': 0.856201171875, 'a_identity': 2.4819580078125, 'b_identity': 7.4580322265625, 'std_low_carbon_preference': 0.2073486328125, 'set_seed': 1}
    base_params = {
    "phi_lower": 1e-2,
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
    "vary_seed_state": "network",
    "alpha_change_state": "dynamic_identity_determined_weights",
    "seed_reps": 25,
    "network_structure_seed": 10, 
    "preferences_seed": 77, 
    "shuffle_homophily_seed": 2,
    "shuffle_coherance_seed": 5,
    "set_seed": 4, 
    "carbon_price_duration": 1000, 
    "burn_in_duration": 0, 
    "N": 200, 
    "M": 2, 
    "sector_substitutability": 2, 
    "low_carbon_substitutability_lower": 2, 
    "low_carbon_substitutability_upper": 2, 
    "a_identity": 2, 
    "b_identity": 2, 
    "clipping_epsilon_init_preference": 1e-5,
    "std_low_carbon_preference": 0.01, 
    "confirmation_bias": 5, 
    "init_carbon_price": 0, 
    "homophily_state": 0,
    "coherance_state": 0.95,
    "BA_nodes": 11,
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
        single_experiment_plot.main(fileName = fileName)
