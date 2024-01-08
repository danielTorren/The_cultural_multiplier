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
    base_params = {
        "save_timeseries_data_state": 1,
        "compression_factor_state": 1,
        "heterogenous_intrasector_preferences_state": 1,
        "heterogenous_carbon_price_state": 0,
        "heterogenous_sector_substitutabilities_state": 0,
        "heterogenous_phi_state": 0,
        "imperfect_learning_state": 0,
        "ratio_preference_or_consumption_state": 0,
        "alpha_change_state": "dynamic_culturally_determined_weights",
        "vary_seed_imperfect_learning_state_or_initial_preferences_state": 0,
        "static_internal_preference_state": 1,
        'homophily_state': 1.0, 
        'network_structure_seed': 8, 
        'init_vals_seed': 14, 
        'set_seed': 4, 
        'carbon_price_duration': 3000, 
        'burn_in_duration': 0, 
        'N': 200, 
        'M': 2, 
        'network_density': 0.1, 
        'prob_rewire': 0.1, 
        'sector_substitutability': 1.5, 
        'low_carbon_substitutability_lower': 1.5, 
        'low_carbon_substitutability_upper': 1.5, 
        'a_identity': 2, 
        'b_identity': 2, 
        'clipping_epsilon': 1e-3, 
        'clipping_epsilon_init_preference': 1e-3,
        'std_low_carbon_preference': 0.01, 
        'std_learning_error': 0.02, 
        'confirmation_bias': 0, 
        'expenditure': 10,
        'init_carbon_price': 0, 
        "carbon_price_increased_lower": 0.5,
        "carbon_price_increased_upper": 0.5,
        'phi_lower': 0.2, 
        'phi_upper': 0.1, 
        }
    
    fileName = main(base_params=base_params)

    RUN_PLOT = 1

    if RUN_PLOT:
        single_experiment_plot.main(fileName = fileName)
