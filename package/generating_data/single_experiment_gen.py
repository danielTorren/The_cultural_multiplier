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
    
    """
    base_params = {#BA
    "carbon_price_increased_lower": 0.1,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "network_type": "BA",
    "heterogenous_intrasector_preferences_state": 1,
    "heterogenous_carbon_price_state": 0,
    "heterogenous_sector_substitutabilities_state": 0,
    "SBM_block_heterogenous_individuals_substitutabilities_state": 0,
    "heterogenous_phi_state": 0,
    "imperfect_learning_state": 1,
    "ratio_preference_or_consumption_state": 0,
    "alpha_change_state": "dynamic_culturally_determined_weights",
    "vary_seed_imperfect_learning_state_or_initial_preferences_state": 1,
    "static_internal_preference_state": 0,
    "seed_reps": 5,
    "network_structure_seed": 8, 
    "init_vals_seed": 14, 
    "set_seed": 4, 
    "carbon_price_duration": 1000, 
    "burn_in_duration": 0, 
    "N": 200, 
    "M": 1, 
    "sector_substitutability": 1.5, 
    "low_carbon_substitutability_lower": 1.5, 
    "low_carbon_substitutability_upper": 1.5, 
    "a_identity": 2, 
    "b_identity": 2, 
    "clipping_epsilon": 1e-5, 
    "clipping_epsilon_init_preference": 1e-5,
    "std_low_carbon_preference": 0.01, 
    "std_learning_error": 0.02, 
    "confirmation_bias": 1, 
    "expenditure": 1,
    "init_carbon_price": 0, 
    "phi_lower": 0.01, 
    "BA_nodes": 11
    }
    """

    #"""
    base_params = {#SBM
    "carbon_price_increased_lower": 0.1,
    "save_timeseries_data_state": 0,
    "compression_factor_state": 1,
    "network_type": "SBM",
    "heterogenous_intrasector_preferences_state": 1,
    "heterogenous_carbon_price_state": 0,
    "heterogenous_sector_substitutabilities_state": 0,
    "SBM_block_heterogenous_individuals_substitutabilities_state": 0,
    "heterogenous_phi_state": 0,
    "imperfect_learning_state": 1,
    "ratio_preference_or_consumption_state": 0,
    "alpha_change_state": "dynamic_culturally_determined_weights",
    "vary_seed_imperfect_learning_state_or_initial_preferences_state": 1,
    "static_internal_preference_state": 0,
    "seed_reps": 5,
    "network_structure_seed": 8, 
    "init_vals_seed": 14, 
    "set_seed": 4, 
    "carbon_price_duration": 1000, 
    "burn_in_duration": 0, 
    "N": 200, 
    "M": 1, 
    "sector_substitutability": 1.5, 
    "low_carbon_substitutability_lower": 1.5, 
    "low_carbon_substitutability_upper": 1.5, 
    "a_identity": 2, 
    "b_identity": 2, 
    "clipping_epsilon": 1e-5, 
    "clipping_epsilon_init_preference": 1e-5,
    "std_low_carbon_preference": 0.01, 
    "std_learning_error": 0.02, 
    "confirmation_bias": 1, 
    "expenditure": 1,
    "init_carbon_price": 0, 
    "phi_lower": 0.01, 
    "SBM_block_num": 2,
    "SBM_network_density_input_intra_block": 0.2,
    "SBM_network_density_input_inter_block": 0.005
    }
    #"""

    base_params["BA_green_or_brown_hegemony"] = 0    
    base_params["homophily_state"] = 0
    base_params["alpha_change_state"] = "fixed_preferences"
    base_params["phi"] = 0 #double sure!
    base_params["seed_reps"] = 1
    
    b = {
    "homophily_state" : 0,
    "carbon_price_increased_lower": 0,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "network_type": "SBM",
    "heterogenous_intrasector_preferences_state": 0,
    "heterogenous_carbon_price_state": 0,
    "heterogenous_sector_substitutabilities_state": 0,
    "SBM_block_heterogenous_individuals_substitutabilities_state": 1,
    "heterogenous_phi_state": 0,
    "imperfect_learning_state": 1,
    "ratio_preference_or_consumption_state": 0,
    "alpha_change_state": "dynamic_culturally_determined_weights",
    "vary_seed_imperfect_learning_state_or_initial_preferences_state": 1,
    "static_internal_preference_state": 0,
    "seed_reps": 5,
    "network_structure_seed": 8, 
    "init_vals_seed": 14, 
    "set_seed": 4, 
    "carbon_price_duration": 10, 
    "burn_in_duration": 0, 
    "N": 50, 
    "M": 2, 
    "sector_substitutability": 1.5, 
    "low_carbon_substitutability_lower": 1.5, 
    "low_carbon_substitutability_upper": 5, 
    "a_identity": 2, 
    "b_identity": 2, 
    "clipping_epsilon": 1e-5, 
    "clipping_epsilon_init_preference": 1e-5,
    "std_low_carbon_preference": 0.01, 
    "std_learning_error": 0.02, 
    "confirmation_bias": 1, 
    "expenditure": 1,
    "init_carbon_price": 0, 
    "phi_lower": 0.01, 
    "SBM_block_num": 2,
    "SBM_network_density_input_intra_block": 0.2,
    "SBM_network_density_input_inter_block": 0.005,
    "SBM_sub_add_on": 1,
    }
    
    """
    a = {
        "save_timeseries_data_state": 1,
        "compression_factor_state": 1,
        "network_type": "SBM",
        "heterogenous_intrasector_preferences_state": 1,
        "heterogenous_carbon_price_state": 0,
        "heterogenous_sector_substitutabilities_state": 0,
        "SBM_block_heterogenous_individuals_substitutabilities_state": 0,
        "heterogenous_phi_state": 0,
        "imperfect_learning_state": 1,
        "ratio_preference_or_consumption_state": 0,
        "alpha_change_state": "dynamic_culturally_determined_weights",
        "vary_seed_imperfect_learning_state_or_initial_preferences_state": 1,
        "static_internal_preference_state": 0,
        'homophily_state': 1, 
        'network_structure_seed': 8, 
        'init_vals_seed': 14, 
        'set_seed': 4, 
        'carbon_price_duration': 10, 
        'burn_in_duration': 0, 
        'N': 200, 
        'M': 2, 
        'sector_substitutability': 1.5, 
        'low_carbon_substitutability_lower': 1.5, 
        'low_carbon_substitutability_upper': 1.5, 
        'a_identity': 2, 
        'b_identity': 2, 
        'clipping_epsilon': 1e-5, 
        'clipping_epsilon_init_preference': 1e-5,
        'std_low_carbon_preference': 0.01, 
        'std_learning_error': 0.02, 
        'confirmation_bias': 1, 
        'expenditure': 1,
        'init_carbon_price': 0, 
        "carbon_price_increased_lower": 0.5,
        "carbon_price_increased_upper": 0.5,
        'phi_lower': 0.01, 
        'phi_upper': 0.01, 
        #NETWORK STUFF
        "SW_network_density": 0.1, 
        "SW_prob_rewire": 0.1,
        "SF_alpha": 0.35,
        "SF_beta":0.3,
        "SF_gamma":0.35,
        "SBM_block_num": 2,
        "SBM_network_density_input_intra_block": 0.2,
        "SBM_network_density_input_inter_block": 0.005,
        "SBM_sub_add_on": 1,
        "BA_nodes": 11,
        "BA_green_or_brown_hegemony": 0
        }
    """
    
    fileName = main(base_params=base_params)

    RUN_PLOT = 1

    if RUN_PLOT:
        single_experiment_plot.main(fileName = fileName)
