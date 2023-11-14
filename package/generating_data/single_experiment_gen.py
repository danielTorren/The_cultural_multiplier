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
    "budget_inequality_state":0,
    "heterogenous_preferences": 1.0,
    "redistribution_state": 0,
    "static_internal_A_state": 0,
    "utility_function_state": "nested_CES",
    "dividend_progressiveness":0,
    "compression_factor":10,
    "carbon_tax_implementation": "flat", 
    "ratio_preference_or_consumption_identity": 1.0,
    "ratio_preference_or_consumption": 0.0,
    "set_seed": 10,
    "network_structure_seed": 1,
    "carbon_price_duration": 2000,
    "burn_in_duration": 0,
    "N": 200,
    "M": 10,
    "phi": 0.025,
    "network_density": 0.1,
    "prob_rewire": 0.1,
    "learning_error_scale": 0.02,
    "homophily": 0.95,
    "confirmation_bias": 10,
    "carbon_price_increased": 0.5,
    "sector_substitutability": 1.5,
    "low_carbon_substitutability_lower":2,
    "low_carbon_substitutability_upper":5,
    "a_identity": 1,
    "b_identity": 1,
    "std_low_carbon_preference": 0.05,
    "init_carbon_price": 0,
    "clipping_epsilon": 1e-4,
    "lambda_m_lower": 1.1,
    "lambda_m_upper": 10
}
    fileName = main(base_params=base_params)

    RUN_PLOT = 1

    if RUN_PLOT:
        single_experiment_plot.main(fileName = fileName)
