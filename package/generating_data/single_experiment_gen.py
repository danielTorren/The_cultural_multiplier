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
    print(Data.average_identity)

    createFolder(fileName)
    save_object(Data, fileName + "/Data", "social_network")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':
    base_params = {
    "save_timeseries_data": 1, 
    "time_steps_max": 2000,
    "carbon_price_time": 1000,
    "phi_lower": 0.01,
    "phi_upper": 0.05,
    "compression_factor": 10,
    "seed_list": [1,2,3,4,5],
    "set_seed": 1,
    "N": 200,
    "M": 3,
    "K": 20,
    "prob_rewire": 0.1,
    "learning_error_scale": 0.02,
    "homophily": 0.95,
    "confirmation_bias": 20,
    "a_low_carbon_preference": 1,
    "b_low_carbon_preference": 1,
    "a_service_preference": 1,
    "b_service_preference": 1,
    "a_low_carbon_substitutability":1,
    "b_low_carbon_substitutability":1,
    "multiplier_low_carbon_substitutability":10,
    "a_individual_budget": 1,
    "b_individual_budget": 1,
    "a_prices_high_carbon": 1,
    "b_prices_high_carbon": 1,
    "service_substitutability": 10,
    "init_carbon_price": 0,
    "dividend_progressiveness": 0.0,
    "carbon_price_increased" : 1.0,
    "budget_multiplier": 1,
    "price_high_carbon_factor": 1.0,
    "ratio_preference_or_consumption": 1.00,
    "carbon_tax_implementation": "flat",
    "clipping_epsilon": 1e-5
}
    fileName = main(base_params=base_params)