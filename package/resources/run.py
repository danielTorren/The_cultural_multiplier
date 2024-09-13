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

    #print("tim step max", parameters["time_steps_max"])
    #quit()
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

def parallel_run(params_dict: dict[dict]) -> list[Network]:
    """
    Generate data from a list of parameter dictionaries, parallelize the execution of each single shot simulation

    """

    num_cores = multiprocessing.cpu_count()
    #data_parallel = [generate_data(i) for i in params_dict]
    data_parallel = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_data)(i) for i in params_dict)
    return data_parallel

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
###################################################################################################
###################################################################################################

def generate_emissions_stock_res(params):
    #print("params", params) 
    #quit()
    data = generate_data(params)
    return data.total_carbon_emissions_stock

def emissions_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #emissions_list = [generate_emissions_stock_res(i) for i in params_dict]
    emissions_list = Parallel(n_jobs=num_cores, verbose=10)(    delayed(generate_emissions_stock_res)(i) for i in params_dict)
    return np.asarray(emissions_list)
###################################################################################################


######################################################################################################################################################################################################

###################################################################################################

###################################################################################################
def generate_identity_timeseries(params):
    data = generate_data(params)
    return data.history_identity_vec

def identity_timeseries_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_identity_timeseries(i) for i in params_dict]
    identity_timeseries_list = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_identity_timeseries)(i) for i in params_dict
    )
    return np.asarray(identity_timeseries_list)

###################################################################################################

def generate_undershoot_timeseries(params):
    data = generate_data(params)
    return data.history_identity_vec, data.history_low_carbon_preference_matrix, data.history_flow_carbon_emissions, data.history_stock_carbon_emissions

def undershoot_timeseries_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_identity_timeseries(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_undershoot_timeseries)(i) for i in params_dict
    )
    identity_timeseries_list, preferences_timeseries_list, flow_list, stock_list = zip(
        *res
    )

    return np.asarray(identity_timeseries_list), np.asarray(preferences_timeseries_list), np.asarray(flow_list), np.asarray(stock_list)

###################################################################################################
def generate_emissions_stock_res_sectors(params):
    data = generate_data(params)
    return data.total_carbon_emissions_stock, data.total_carbon_emissions_stock_sectors

def emissions_parallel_run_sectors(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #emissions_list = [generate_emissions_stock_res(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_emissions_stock_res_sectors)(i) for i in params_dict
    )
    emissions_list, emissions_list_sectors = zip(
        *res
    )
    return np.asarray(emissions_list), np.asarray(emissions_list_sectors)

###################################################################################################

def generate_emissions_stock_res_timeseries(params):
    data = generate_data(params)
    return data.history_stock_carbon_emissions

def emissions_parallel_run_timeseries(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #emissions_list = [generate_emissions_stock_res_timeseries(i) for i in params_dict]
    emissions_list = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_emissions_stock_res_timeseries)(i) for i in params_dict
    )
    return np.asarray(emissions_list)

################################################################################
#Bifurcation of preferences with 1d param vary
def generate_preferences_res(params):
    #get out the N by M matrix of final preferences
    data = generate_data(params)

    data_individual_preferences = []

    for v in range(data.N):
        data_individual_preferences.append(np.asarray(data.agent_list[v].low_carbon_preferences))

    return np.asarray(data_individual_preferences)


def preferences_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_stock_res(i) for i in params_dict]
    preferences_array_list = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_preferences_res)(i) for i in params_dict
    )
    return np.asarray(preferences_array_list)#shaep is #of runs, N indiviuduals, M preferences

def generate_preferences_consumption_res(params):
    #get out the N by M matrix of final preferences
    data = generate_data(params)

    data_individual_preferences = []
    data_individual_H = []
    data_individual_L = []

    for v in range(data.N):
        data_individual_preferences.append(np.asarray(data.agent_list[v].low_carbon_preferences))
        data_individual_H.append(np.asarray(data.agent_list[v].H_m))
        data_individual_L.append(np.asarray(data.agent_list[v].L_m))

    return np.asarray(data_individual_preferences),np.asarray(data_individual_H),np.asarray(data_individual_L)

def preferences_consumption_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #res = [generate_emissions_stock_res(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_preferences_consumption_res)(i) for i in params_dict
    )
    preferences, high_carbon, low_carbon= zip(
        *res
    )
    return np.asarray(preferences),np.asarray(high_carbon),np.asarray(low_carbon)
#shaep is #of runs, N indiviuduals, M preferences

##############################################################################


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
    res = [generate_multi_output_individual_emissions_list(i) for i in params_dict]
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
    #emissions_stock = [generate_emissions_stock(i) for i in params_dict]
    emissions_stock = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_emissions_stock)(i) for i in params_dict
    )


    return np.asarray(emissions_stock)

def generate_emissions_stock_flow_end(params):
    data = generate_data(params)
    norm = params["N"]*params["M"]
    return np.asarray(data.total_carbon_emissions_stock/norm), np.asarray(data.total_carbon_emissions_flow/norm)
    #return np.asarray(data.total_carbon_emissions_stock), np.asarray(data.total_carbon_emissions_flow)

def multi_emissions_stock_flow_end(
        params_dict: list[dict]
) -> npt.NDArray:
    
    #res = [generate_emissions_stock_flow_end(i) for i in params_dict]
    num_cores = multiprocessing.cpu_count()
    res = Parallel(n_jobs=num_cores, verbose=10)(   delayed(generate_emissions_stock_flow_end)(i) for i in params_dict)
    emissions_stock, emissions_flow = zip(
        *res
    )

    return np.asarray(emissions_stock), np.asarray(emissions_flow)

