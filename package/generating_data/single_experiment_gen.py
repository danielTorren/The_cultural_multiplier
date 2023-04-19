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

    createFolder(fileName)
    save_object(Data, fileName + "/Data", "social_network")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':
    base_params = {
    "save_timeseries_data": 1, 
    "alpha_change" : "dynamic_culturally_determined_weights",
    "time_steps_max": 100,
    "phi_lower": 0.01,
    "phi_upper": 0.05,
    "compression_factor": 10,
    "set_seed": 2,
    "N": 50,
    "M": 3,
    "K": 5,
    "prob_rewire": 0.1,
    "cultural_inertia": 1000,
    "learning_error_scale": 0.02,
    "discount_factor": 0.95,
    "homophily": 0.95,
    "confirmation_bias": 40,
    "a_attitude": 1,
    "b_attitude": 1,
    "a_threshold": 1,
    "b_threshold": 1,
    "green_N": 0,
}
    fileName = main(base_params=base_params)