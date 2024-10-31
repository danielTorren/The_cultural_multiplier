"""Run simulation 
A module that use input data to run the simulation for a given number of timesteps.
Multiple simulations at once in parallel can also be run. 



Created: 10/10/2022
"""

# imports
import time
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
import multiprocessing
from package.model.network_matrix import Network_Matrix as Network

# modules
####SINGLE SHOT RUN
def generate_data(parameters: dict,print_simu = 0) -> Network:
    """
    Generate the Network object which itself contains list of Individual objects. Run this forward in time for the desired number of steps

    Parameters
    ----------
    parameters: dict
        Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

    Returns
    -------
    social_network: Network
        Social network that has evolved from initial conditions
    """

    if print_simu:
        start_time = time.time()

    parameters["time_steps_max"] = parameters["burn_in_duration"] + parameters["carbon_price_duration"]

    social_network = Network(parameters)

    #### RUN TIME STEPS
    while social_network.t < parameters["time_steps_max"]:
        social_network.next_step()
        #print("step", social_network.t)

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    return social_network
###################################################################################################

def generate_sensitivity_output_flat(params: dict):
    data = generate_data(params)
    return data.total_carbon_emissions_stock

def parallel_run_sa(
    params_dict: list[dict],
):

    num_cores = multiprocessing.cpu_count()
    #results_emissions_stock =[generate_sensitivity_output_flat(i) for i in params_dict]
    results_emissions_stock = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_sensitivity_output_flat)(i) for i in params_dict)
    return np.asarray(results_emissions_stock)

###################################################################################################


def generate_emissions_stock_res(params):
    data = generate_data(params)
    return data.total_carbon_emissions_stock

def emissions_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #emissions_list = [generate_emissions_stock_res(i) for i in params_dict]
    emissions_list = Parallel(n_jobs=num_cores, verbose=10)(    delayed(generate_emissions_stock_res)(i) for i in params_dict)
    return np.asarray(emissions_list)

##############################################################################

def generate_emissions_stock(params):
    data = generate_data(params)
    return np.asarray(data.total_carbon_emissions_stock)

def multi_emissions_stock(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #emissions_stock = [generate_emissions_stock(i) for i in params_dict]
    emissions_stock = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_emissions_stock)(i) for i in params_dict
    )


    return np.asarray(emissions_stock)
##############################################################################


