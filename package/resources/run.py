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
from package.model.network import Network

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

    social_network = Network(parameters)

    #### RUN TIME STEPS
    while social_network.t < parameters["time_steps_max"]:
        social_network.next_step()

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    return social_network

def generate_first_behaviour_lists_one_seed_output(params):
    """For birfurcation just need attitude of first behaviour"""
    data = generate_data(params)
    return [x.attitudes[0] for x in data.agent_list]

def generate_multi_output_individual_emissions_list(params):
    """Individual specific emission and associated id to compare runs with and without behavioural interdependence"""

    emissions_list = []
    carbon_emissions_not_influencer = []
    for v in params["seed_list"]:
        params["set_seed"] = v
        data = generate_data(params)
        emissions_list.append(data.total_carbon_emissions)
        carbon_emissions_not_influencer.append(sum(x.total_carbon_emissions for x in data.agent_list if not x.green_fountain_state))
    return (emissions_list, carbon_emissions_not_influencer)

def generate_sensitivity_output(params: dict):
    """
    Generate data from a set of parameter contained in a dictionary. Average results over multiple stochastic seeds contained in params["seed_list"]

    """

    emissions_list = []
    mean_list = []
    var_list = []
    coefficient_variance_list = []
    emissions_change_list = []

    for v in params["seed_list"]:
        params["set_seed"] = v
        data = generate_data(params)
        norm_factor = data.N * data.M
        # Insert more measures below that want to be used for evaluating the
        emissions_list.append(data.total_carbon_emissions / norm_factor)
        mean_list.append(data.average_identity)
        var_list.append(data.var_identity)
        coefficient_variance_list.append(data.std_identity / (data.average_identity))
        emissions_change_list.append(np.abs(data.total_carbon_emissions - data.init_total_carbon_emissions)/norm_factor)

    stochastic_norm_emissions = np.mean(emissions_list)
    stochastic_norm_mean = np.mean(mean_list)
    stochastic_norm_var = np.mean(var_list)
    stochastic_norm_coefficient_variance = np.mean(coefficient_variance_list)
    stochastic_norm_emissions_change = np.mean(emissions_change_list)

    return (
        stochastic_norm_emissions,
        stochastic_norm_mean,
        stochastic_norm_var,
        stochastic_norm_coefficient_variance,
        stochastic_norm_emissions_change
    )

def parallel_run(params_dict: dict[dict]) -> list[Network]:
    """
    Generate data from a list of parameter dictionaries, parallelize the execution of each single shot simulation

    """

    num_cores = multiprocessing.cpu_count()
    #data_parallel = [generate_data(i) for i in params_dict]
    data_parallel = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_data)(i) for i in params_dict
    )
    return data_parallel

def multi_stochstic_emissions_run_all_individual(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_multi_output_individual_emissions_list(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_multi_output_individual_emissions_list)(i) for i in params_dict
    )
    emissions_list, carbon_emissions_not_influencer = zip(
        *res
    )

    return np.asarray(emissions_list),np.asarray(carbon_emissions_not_influencer)

def one_seed_identity_data_run(
        params_dict: list[dict]
) -> npt.NDArray:

    num_cores = multiprocessing.cpu_count()
    #res = [generate_sensitivity_output(i) for i in params_dict]
    results_identity_lists = Parallel(n_jobs=num_cores, verbose=10)(

        delayed(generate_first_behaviour_lists_one_seed_output)(i) for i in params_dict
    )

    return np.asarray(results_identity_lists)#can't run with multiple different network sizes


def parallel_run_sa(
    params_dict: dict[dict],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Generate data for sensitivity analysis for model varying lots of parameters dictated by params_dict, producing output
    measures emissions,mean,variance and coefficient of variance. Results averaged over multiple runs with different stochastic seed

    """

    #print("params_dict", params_dict)
    num_cores = multiprocessing.cpu_count()
    #res = [generate_sensitivity_output(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_sensitivity_output)(i) for i in params_dict
    )
    results_emissions, results_mean, results_var, results_coefficient_variance, results_emissions_change = zip(
        *res
    )

    return (
        np.asarray(results_emissions),
        np.asarray(results_mean),
        np.asarray(results_var),
        np.asarray(results_coefficient_variance),
        np.asarray(results_emissions_change)
    )
