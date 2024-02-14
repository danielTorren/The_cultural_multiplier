
# imports
from ast import arg
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_stock,generate_data
from package.resources.utility import produce_param_list_stochastic
from package.generating_data.static_preferences_emissions_gen import calculate_emissions
from package.plotting_data import BA_SBM_plot
from package.resources.utility import generate_vals
##################################################################################################
#REVERSE Engineer the carbon price based on the final emissions

def objective_function(P_H, *args):

    emissions_target, t_max, B, N, M, a, P_L, A, sigma, nu = args

    E = calculate_emissions(t_max, B, N, M, a, P_L, P_H, A, sigma, nu)
    #print("P_H",P_H)
    #print("E state:",emissions_target, E, emissions_target - E )
    convergence_val = (emissions_target - E )#*1e9
    
    return convergence_val

def newton_raphson_with_bounds(func, initial_guess, args_E,bounds = None, tol=1e-6, max_iter=100, delta=1e-7):
    """
    Newton-Raphson method for finding a root of a function without an explicit derivative with bounds.

    Parameters:
    - func: The function for which we want to find the root.
    - initial_guess: Initial guess for the root.
    - bounds: Tuple (lower_bound, upper_bound) for the root.
    - tol: Tolerance, stopping criterion based on the absolute change in the root.
    - max_iter: Maximum number of iterations.
    - delta: Small value for numerical differentiation.

    Returns:
    - root: The approximate root found by the method.
    - iterations: The number of iterations performed.
    """

    root = initial_guess
    iterations = 0

    while abs(func(root, *args_E)) > tol and iterations < max_iter:
        # Numerical approximation of the derivative
        derivative_approx = (func(root + delta, *args_E) - func(root - delta, *args_E)) / (2 * delta)

        # Update the root using Newton-Raphson formula
        root = root - func(root, *args_E) / derivative_approx

        # Check if the updated root is within bounds
        if bounds is not None:
            root = max(min(root, bounds[1]), bounds[0])

        iterations += 1

    return root, iterations

def optimize_PH(emissions_val, t_max, B, N, M, a, P_L, A, sigma, nu, initial_guess):

    args_E = (emissions_val,t_max, B, N, M, a, P_L, A, sigma, nu)
    bounds = (0.1, 10)

    root, iterations = newton_raphson_with_bounds(objective_function,initial_guess = initial_guess, args_E = args_E, bounds = bounds,tol=1e-6, max_iter=100, delta=1e-7)

    return root

def calc_tau_static_preference(emissions_min, emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, initial_guess_min, initial_guess_max):
    
    min_P_H = optimize_PH(emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, initial_guess_min)
    max_P_H = optimize_PH(emissions_min, t_max, B, N, M, a, P_L, A, sigma, nu, initial_guess_max)
    min_tau = min_P_H - 1
    max_tau = max_P_H - 1
    return min_tau, max_tau

def main(
        BASE_PARAMS_LOAD_BA = "package/constants/base_params.json",
        BASE_PARAMS_LOAD_SBM = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
         ) -> str: 

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
    property_reps = var_params["property_reps"]#10,

    property_values_list = generate_vals(
        var_params
    )
    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD_BA)
    params_BA = json.load(f)

    f = open(BASE_PARAMS_LOAD_SBM)
    params_SBM = json.load(f)

    root = "BA_SBM_tau_vary"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)


#########################################################################################################
    #NO HEGEMONOY AND COMPLETE MIXING
    params_BA["BA_green_or_brown_hegemony"] = 0    
    params_BA["homophily_state"] = 0
    params_list_no_heg_BA = produce_param_list_stochastic(params_BA, property_values_list, property_varied)

    #Green HEGEMONOY AND homophily
    params_BA["BA_green_or_brown_hegemony"] = 1   
    params_BA["homophily_state"] = 1
    params_list_green_heg_BA = produce_param_list_stochastic(params_BA, property_values_list, property_varied)

    #Brown HEGEMONOY AND homophily
    params_BA["BA_green_or_brown_hegemony"] = -1   
    params_BA["homophily_state"] = 1
    params_list_brown_heg_BA = produce_param_list_stochastic(params_BA, property_values_list, property_varied)

    params_list_BA = params_list_no_heg_BA + params_list_green_heg_BA + params_list_brown_heg_BA
    

##########################################################################################################
    
    #NO "homophily_state"
    params_SBM["homophily_state"] = 0    
    params_list_no_tau_SBM = produce_param_list_stochastic(params_SBM, property_values_list, property_varied)

    #Low "homophily_state"
    params_SBM["homophily_state"] = 0.5   
    params_list_low_tau_SBM = produce_param_list_stochastic(params_SBM, property_values_list, property_varied)

    #High "homophily_state"
    params_SBM["homophily_state"] = 1  
    params_list_high_tau_SBM = produce_param_list_stochastic(params_SBM, property_values_list, property_varied)

    params_list_SBM = params_list_no_tau_SBM + params_list_low_tau_SBM + params_list_high_tau_SBM
    
    

############################################################################################################################
    #RUN THE STUFF
    params_list = params_list_BA + params_list_SBM
    print("TOTAL RUNS", len(params_list))

    emissions_stock_serial = multi_emissions_stock(params_list)
    emissions_array = emissions_stock_serial.reshape(2, 3, property_reps, params_BA["seed_reps"])#2 is for BA and SBM,3 is for the 3 differents states
    emissions_array_BA = emissions_array[0]
    emissions_array_SBM = emissions_array[1]

    #SAVE STUFF
    createFolder(fileName)
    
    save_object(emissions_array_BA , fileName + "/Data", "emissions_array_BA")
    save_object(emissions_array_SBM, fileName + "/Data", "emissions_array_SBM")
    save_object(params_BA, fileName + "/Data", "base_params_BA")
    save_object(params_SBM, fileName + "/Data", "base_params_SBM")

    print("RUNS DONE")
########################################################################################################################
    t_max = params_SBM["carbon_price_duration"] + params_SBM["burn_in_duration"]

#REFERENCE CASE, helps with the base values
    params_SBM["carbon_price_increased"] = 0#this doesnt matter but needs to be set
    params_SBM["carbon_price_duration"] = 0#just make them begin
    reference_run = generate_data(params_SBM)  # run the simulation
    save_object(reference_run, fileName + "/Data", "reference_run")

    B, N, M, a, P_L, A, sigma, nu= (
        params_SBM["expenditure"],
        params_SBM["N"],
        params_SBM["M"],
        reference_run.sector_preferences,
        reference_run.prices_low_carbon,
        reference_run.low_carbon_preference_matrix_init,
        np.asarray(reference_run.low_carbon_substitutability_array_list),
        reference_run.sector_substitutability
        )
    ###################################################################

    #calc the fixed emissions
    emissions_min =  np.min([np.min(emissions_array_BA),np.min(emissions_array_SBM)])
    emissions_max = np.max([np.max(emissions_array_BA),np.max(emissions_array_SBM)])
    #print("E Min",emissions_min)
    #print("E Max",emissions_max)
    #THESE ARE THE BOUNDS OF WHAT I NEED TO ACHIEVE, I ASSUME THAT BECAUSE THE FUNCTION IS MONOTONICALLY DECREASING IN TAu then emissions between max and min value are interior?
    initial_guess_min_P_H = 1 + min(property_values_list)
    initial_guess_max_P_H = 1 + max(property_values_list)
    min_tau, max_tau = calc_tau_static_preference(emissions_min, emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, initial_guess_min_P_H, initial_guess_max_P_H)
    print("OPTIMISATION DONE")
    #no calc the emissions over the appropriate range
    total_tau_range_static = np.linspace(min_tau, max_tau, 1000)

    emissions_array_static = [calculate_emissions(t_max, B, N, M, a, P_L, (1 + tau), A, sigma, nu) for tau in property_values_list]
    emissions_array_static_full = [calculate_emissions(t_max, B, N, M, a, P_L, (1 + tau), A, sigma, nu) for tau in total_tau_range_static]

    save_object(min_tau, fileName + "/Data", "min_tau")
    save_object(max_tau, fileName + "/Data", "max_tau")
    save_object(total_tau_range_static, fileName + "/Data", "total_tau_range_static")
    save_object(emissions_array_static, fileName + "/Data", "emissions_array_static")
    save_object(emissions_array_static_full, fileName + "/Data", "emissions_array_static_full")
    print("STATIC DONE")

###############################################################################################################
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD_BA = "package/constants/base_params_BA_tau.json",
        BASE_PARAMS_LOAD_SBM = "package/constants/base_params_SBM_tau.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_BA_SBM_tau.json",
    )
    RUN_PLOT = 0

    if RUN_PLOT:
        BA_SBM_plot.main(fileName = fileName_Figure_1)
