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
    "save_timeseries_data": 1, 
    "alpha_change": "dynamic_culturally_determined_weights",
    "heterogenous_preferences": 1.0,
    "compression_factor":2,
    "ratio_preference_or_consumption":0.0,
    "network_structure_seed": 8,
    "init_vals_seed": 14,
    "set_seed": 4,
    "carbon_price_duration": 2000,
    "burn_in_duration": 0,
    "N": 200,
    "M": 4,
    "phi": 0.005,
    "network_density": 0.1,
    "prob_rewire": 0.1,
    "learning_error_scale": 0.0,
    "homophily": 0.95,
    "init_carbon_price": 0,
    "carbon_price_increased": 0.2,
    "sector_substitutability": 5,
    "low_carbon_substitutability_lower":5,
    "low_carbon_substitutability_upper":5,
    "a_identity": 2,
    "b_identity": 2,
    "clipping_epsilon": 1e-5,
    "std_low_carbon_preference":0.01,
    "confirmation_bias":10,
    "budget":100
}
    fileName = main(base_params=base_params)

    RUN_PLOT = 1

    if RUN_PLOT:
        single_experiment_plot.main(fileName = fileName)
