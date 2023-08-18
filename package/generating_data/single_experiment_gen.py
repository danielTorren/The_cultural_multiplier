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

def main(
    base_params
) -> str: 

    root = "single_experiment"
    fileName = produce_name_datetime(root)
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
    "budget_inequality_state":0,
    "heterogenous_preferences": 1.0,
    "redistribution_state": 0,
    "alpha_change": "static_culturally_determined_weights",
    "utility_function_state": "nested_CES",
    "dividend_progressiveness":0,
    "compression_factor":10,
    "carbon_price_duration": 1,
    "burn_in_duration": 0,
    "phi": 0.05,
    "set_seed": 1,
    "N": 1000,
    "M": 2,
    "network_density": 0.3,
    "prob_rewire": 0.1,
    "learning_error_scale": 0.02,
    "homophily": 0.95,
    "confirmation_bias": 0,
    "service_substitutability": 2,
    "low_carbon_substitutability_lower":2,
    "low_carbon_substitutability_upper":5,
    "lambda_m_lower": 1.1,
    "lambda_m_upper": 10,
    "a_identity": 2,
    "b_identity": 2,
    "var_low_carbon_preference": 0.03,
    "init_carbon_price": 0,
    "carbon_price_increased":0.5,
    "clipping_epsilon": 1e-4,
    "carbon_tax_implementation": "flat", 
    "ratio_preference_or_consumption_identity": 1.0,
    "ratio_preference_or_consumption": 0.0
}
    fileName = main(base_params=base_params)