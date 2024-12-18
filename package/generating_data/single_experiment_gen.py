"""Runs a single simulation to produce data which is saved

A module that use dictionary of data for the simulation run. The single shot simulztion is run
for a given initial set seed.

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
    print("EM:", Data.total_carbon_emissions_stock)

    createFolder(fileName)
    save_object(Data, fileName + "/Data", "social_network")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':
    
    base_params = {
    "phi": 0.02,
    "network_type": "SBM",
    "carbon_price_increased": 0.5,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "alpha_change_state": "dynamic_identity_determined_weights",#"dynamic_hybrid_determined_weights",#"fixed_preferences",#"dynamic_socially_determined_weights",#"dynamic_identity_determined_weights"
    "seed_reps": 1,
    "network_structure_seed": 2, 
    "preferences_seed": 99, 
    "shuffle_homophily_seed": 55,
    "shuffle_coherance_seed": 31,
    "carbon_price_duration": 360,#360, 
    "burn_in_duration": 0, 
    "N": 300,#3000, 
    "M": 1, 
    "M_identity": 1,
    "sector_substitutability": 2, 
    "low_carbon_substitutability": 4, 
    "a_preferences": 2, 
    "b_preferences": 2, 
    "clipping_epsilon_init_preference": 1e-5, 
    "confirmation_bias": 5, 
    "init_carbon_price": 0, 
    "homophily_state": 1,
    "coherance_state": 0.9,
    "SF_density":0.1,
    "SF_green_or_brown_hegemony": -1,
    "SBM_block_num": 2,
    "SBM_network_density_input_intra_block": 0.2,
    "SBM_network_density_input_inter_block": 0.005,
    "SW_network_density": 0.1,#0.1,
    "SW_prob_rewire": 0.1
    }

    fileName = main(base_params=base_params)

    RUN_PLOT = 1

    if RUN_PLOT:
        single_experiment_plot.main(fileName = fileName)
