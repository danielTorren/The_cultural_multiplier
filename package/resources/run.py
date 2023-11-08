"""Run simulation 
A module that use input data to run the simulation for a given number of timesteps.
Multiple simulations at once in parallel can also be run. 



Created: 10/10/2022
"""

# imports
from gc import callbacks
import time
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
import multiprocessing
from package.model.network import Network
#from scipy.optimize import least_squares
from copy import deepcopy
from scipy.optimize import minimize, NonlinearConstraint


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

    #print("tim step max", parameters["time_steps_max"],parameters["burn_in_duration"], parameters["carbon_price_duration"])
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


def generate_emissions_stock_res(params):
    data = generate_data(params)
    return data.total_carbon_emissions_stock

def emissions_parallel_run(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #emissions_list = [generate_emissions_stock_res(i) for i in params_dict]
    emissions_list = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_emissions_stock_res)(i) for i in params_dict
    )
    return np.asarray(emissions_list)

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
    norm = params["N"]*params["M"]
    return np.asarray(data.total_carbon_emissions_stock/norm), np.asarray(data.init_total_carbon_emissions/norm)

def multi_emissions_stock(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #emissions_stock = [generate_emissions_stock(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_emissions_stock)(i) for i in params_dict
    )
    emissions_stock, emissions_stock_init = zip(
        *res
    )

    return np.asarray(emissions_stock), np.asarray(emissions_stock_init)

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

def generate_emissions_stock_ineq(params):
    data = generate_data(params)
    norm = params["N"]*params["M"]
    return np.asarray(data.total_carbon_emissions_stock/norm) , data.gini

def multi_emissions_stock_ineq(
        params_dict: list[dict]
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #emissions_stock = [generate_emissions_stock(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_emissions_stock_ineq)(i) for i in params_dict
    )
    emissions_stock, gini_list= zip(
        *res
    )

    return np.asarray(emissions_stock), np.asarray(gini_list)


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

#Running emissions reductions simulations
#################################################################################################################################
#Run the burn in periods
def generate_burn_in_societies(params):
    data = generate_data(params)
    return data

def multi_burn_in_societies(
        params_dict: list[dict]
) -> npt.NDArray:
    #societies = [generate_burn_in_societies(i) for i in params_dict]
    num_cores = multiprocessing.cpu_count()
    societies = Parallel(n_jobs=num_cores)(#, verbose=10
        delayed(generate_burn_in_societies)(i) for i in params_dict
    )

    return np.asarray(societies)

#################################################################################################################################
#run the models with a carbon price (can be zero calc the baseline emissions for a given burn in set up)
def generate_data_load(social_network) -> Network:
    """
    Load model and run it

    """
    social_network.time_steps_max = social_network.burn_in_duration + social_network.carbon_price_duration

    #### RUN TIME STEPS
    while social_network.t <= social_network.time_steps_max:
        social_network.next_step()

    return social_network

def generate_emissions_load(model_burn_in):
    #print("before social_network.time_steps_max ",model_burn_in.time_steps_max,model_burn_in.burn_in_duration, model_burn_in.carbon_price_duration )
    #print("BEFORE norm emissions init",model_burn_in.total_carbon_emissions_stock,model_burn_in.total_carbon_emissions_stock/(model_burn_in.N*model_burn_in.M))
    data = generate_data_load(model_burn_in)
    #print("AFTER norm emissions init",data.total_carbon_emissions_stock,data.total_carbon_emissions_stock/(data.N*data.M))
    #norm = data.N*data.M
    return data.total_carbon_emissions_stock

def multi_emissions_load(
        model_list: list
) -> npt.NDArray:
    num_cores = multiprocessing.cpu_count()
    #emissions_stock = [generate_emissions_stock(i) for i in params_dict]
    emissions_stock = Parallel(n_jobs=num_cores, verbose=10)(#
        delayed(generate_emissions_load)(i) for i in model_list
    )

    return np.asarray(emissions_stock)

######################################################################
#################################################################################################

#Finding carbon price reduction for a given target
def calc_root_emissions_target_load(x, model):
    model_copy = deepcopy(model)
    #print("current emissions, target", model_copy.total_carbon_emissions_stock, model_copy.emissions_stock_target)
    #print("x",x)
    #model_copy.carbon_price_increased = x# i dont know why x is given as a list, and i dont know why is works?
    model_copy.carbon_price_increased = x[0]# i dont know why x is given as a list, and i dont know why is works?
    model_copy_end = generate_data_load(model_copy)
    #norm = model_copy_end.N*model_copy_end.M
    #print("emissiosn after run,target, price",model_copy_end.total_carbon_emissions_stock, model_copy_end.emissions_stock_target,x[0])
    root = model_copy_end.emissions_stock_target - model_copy_end.total_carbon_emissions_stock
    return root

def generate_target_tau_val_load(model,tau_guess):

    # Create a NonlinearConstraint with additional parameters using a lambda function
    #constraint_func = 
    # Create a NonlinearConstraint for the positivity constraint, #Want the emissions to be lower than the target
    positivity_constraint = NonlinearConstraint(lambda x: constraint(x, model), lb=0, ub=np.inf)
    # Call minimize with the constraint
    #
    result = minimize(calc_root_emissions_target_load, x0 = tau_guess, args = (model),bounds=[(0, np.inf)], method='trust-constr',constraints=positivity_constraint)#,
    #result = minimize(calc_root_emissions_target_load, x0 = tau_guess, args = (model), method='trust-constr',constraints=positivity_constraint)#,bounds=(0, np.inf)
    #result = least_squares(lambda x: calc_root_emissions_target_load(x, model),verbose = 1, x0=tau_guess, bounds = (0, np.inf))# xtol=tau_xtol
    #print("result",result)
    #tau_val = result["x"][0]
    tau_val = result.x[0]
    return tau_val

def constraint(x, model):
    # Ensure that the function is positive at x
    return calc_root_emissions_target_load(x,model)


def multi_target_emissions_load(        
        models_matrix,tau_guess
) -> npt.NDArray:
    
    num_cores = multiprocessing.cpu_count()
    
    tau_vals = [generate_target_tau_val_load(i,tau_guess) for i in models_matrix]
    #tau_vals = Parallel(n_jobs=num_cores, verbose=10)(delayed(generate_target_tau_val_load)(i,tau_guess) for i in models_matrix)
    return np.asarray(tau_vals)

########################################################
# Create a class to store the best_model
class BestModelContainer:
    def __init__(self):
        self.best_model = None
        self.termination_flag = False

# Run model changing the carbon price to get to a target:
def optimizing_tax_to_reach_emissions_target(model, tau_guess, stock_target_convergence):
    # Initialize the BestModelContainer
    best_model_container = BestModelContainer()

    # Create a NonlinearConstraint for the positivity constraint
    positivity_constraint = NonlinearConstraint(lambda x: constraint_return_model(x, model,best_model_container), lb=0, ub=np.inf)

    def callback_func(xk,fun):
        #print("xk,fun",xk,fun["fun"])
        if abs(fun["fun"]) < stock_target_convergence: 
            return True
        else:
            return False
        
    # Call minimize with the constraint and callback
    result = minimize(calc_root_emissions_target_load_return_model, x0=tau_guess, args=(model,best_model_container), bounds=[(0, np.inf)], method='trust-constr', constraints=positivity_constraint, callback=callback_func)
    #result = minimize(calc_root_emissions_target_load_return_model, x0=tau_guess, args=(model,best_model_container), bounds=[(0, np.inf)], method='trust-constr', callback=callback_func)
    tau_val = result.x[0]

    return best_model_container.best_model, tau_val

# The rest of your functions remain the same

def calc_root_emissions_target_load_return_model(x, model,best_model_container):
    model_copy = deepcopy(model)
    model_copy.carbon_price_increased = x[0]
    model_copy_end = generate_data_load(model_copy)
    root = model_copy_end.emissions_stock_target - model_copy_end.total_carbon_emissions_stock

    if best_model_container.best_model is None or root < best_model_container.best_model['fun']:
        result = {'x': x, 'fun': root, 'model': model}
        best_model_container.best_model = deepcopy(result)

    print("root checking, x",model_copy_end.emissions_stock_target, model_copy_end.total_carbon_emissions_stock, root, x)
    
    return root

def constraint_return_model(x, model,best_model_container):
    # Ensure that the function is positive at x
    constraint = calc_root_emissions_target_load_return_model(x, model,best_model_container)
    return constraint