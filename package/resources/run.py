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

def generate_sensitivity_output(params: dict):
    """
    Generate data from a set of parameter contained in a dictionary. Average results over multiple stochastic seeds 

    """
    #print("params", params)

    emissions_stock_list = []
    emissions_flow_list = []
    mean_list = []
    var_list = []
    coefficient_variance_list = []
    emissions_change_list = []

    for v in range(params["seed_reps"]):
        params["set_seed"] = int(v+1)#plus one is because seed 0 and 1 are the same, so want to avoid them 
        data = generate_data(params)
        norm_factor = data.N * data.M
        # Insert more measures below that want to be used for evaluating the
        emissions_stock_list.append(data.total_carbon_emissions_stock/norm_factor)
        emissions_flow_list.append(data.total_carbon_emissions_flow)
        mean_list.append(data.average_identity)
        var_list.append(data.var_identity)
        coefficient_variance_list.append(data.std_identity / (data.average_identity))
        emissions_change_list.append(np.abs(data.total_carbon_emissions_stock - data.init_total_carbon_emissions)/norm_factor)

    stochastic_norm_emissions_stock = np.mean(emissions_stock_list)
    stochastic_norm_emissions_flow = np.mean(emissions_flow_list)
    stochastic_norm_mean = np.mean(mean_list)
    stochastic_norm_var = np.mean(var_list)
    stochastic_norm_coefficient_variance = np.mean(coefficient_variance_list)
    stochastic_norm_emissions_change = np.mean(emissions_change_list)

    #print("outputs",         stochastic_norm_emissions_stock,
    #    stochastic_norm_emissions_flow,
    #    stochastic_norm_mean,
    #    stochastic_norm_var,
    #    stochastic_norm_coefficient_variance,
    #    stochastic_norm_emissions_change
    #    )

    return (
        stochastic_norm_emissions_stock,
        stochastic_norm_emissions_flow,
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
    results_emissions_stock, results_emissions_flow, results_mean, results_var, results_coefficient_variance, results_emissions_change = zip(
        *res
    )

    return (
        np.asarray(results_emissions_stock),
        np.asarray(results_emissions_flow),
        np.asarray(results_mean),
        np.asarray(results_var),
        np.asarray(results_coefficient_variance),
        np.asarray(results_emissions_change)
    )

def generate_multi_output_individual_emissions_list(params):
    emissions_list = []
    for v in range(params["seed_reps"]):
        params["set_seed"] = int(v+1)
        data = generate_data(params)
        emissions_list.append(data.total_carbon_emissions_stock)#LOOK AT STOCK
    return (emissions_list)

def multi_stochstic_emissions_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_multi_output_individual_emissions_list(i) for i in params_dict]
    emissions_list = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_multi_output_individual_emissions_list)(i) for i in params_dict
    )
    return np.asarray(emissions_list)



def stochastic_generate_emissions(params):
    data = generate_data(params)
    return data.history_stock_carbon_emissions

def sweep_stochstic_emissions_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_multi_output_individual_emissions_list(i) for i in params_dict]
    emissions_list = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(stochastic_generate_emissions)(i) for i in params_dict
    )
    return np.asarray(emissions_list)

def stochastic_generate_emissions_stock_flow(params):
    emissions_stock_list = []
    emissions_flow_list = []
    for v in range(params["seed_reps"]):
        params["set_seed"] = int(v+1)
        data = generate_data(params)
        emissions_stock_list.append(data.history_stock_carbon_emissions)#LOOK AT STOCK
        emissions_flow_list.append(data.history_flow_carbon_emissions)#LOOK AT STOCK
    return (np.asarray(emissions_stock_list), np.asarray(emissions_flow_list))

def multi_stochstic_emissions_flow_stock_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_multi_output_individual_emissions_list(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(stochastic_generate_emissions_stock_flow)(i) for i in params_dict
    )
    emissions_stock, emissions_flow = zip(
        *res
    )

    return np.asarray(emissions_stock), np.asarray(emissions_flow)

def generate_emissions_stock_flow(params):
    data = generate_data(params)
    return (np.asarray(data.history_stock_carbon_emissions), np.asarray(data.history_flow_carbon_emissions), np.asarray(data.history_identity_list))

def multi_emissions_flow_stock_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_multi_output_individual_emissions_list(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_emissions_stock_flow)(i) for i in params_dict
    )
    emissions_stock, emissions_flow, identity= zip(
        *res
    )

    return np.asarray(emissions_stock), np.asarray(emissions_flow), np.asarray(identity)

def generate_emissions_stock(params):
    data = generate_data(params)
    return np.asarray(data.total_carbon_emissions_stock) 

def multi_emissions_stock(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_multi_output_individual_emissions_list(i) for i in params_dict]
    emissions_stock = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_emissions_stock)(i) for i in params_dict
    )

    return np.asarray(emissions_stock)