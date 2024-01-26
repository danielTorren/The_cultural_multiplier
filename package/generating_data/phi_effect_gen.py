
# imports
from ast import arg
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_stock,generate_data
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list_stochastic
from package.generating_data.static_preferences_emissions_gen import calculate_emissions
from package.plotting_data import phi_effect_plot

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(np.log10(variable_parameters_dict["property_min"]),np.log10( variable_parameters_dict["property_max"]), variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_values_list 
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
        BASE_PARAMS_LOAD_SW = "package/constants/base_params.json",
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

    f = open(BASE_PARAMS_LOAD_SW)
    params_SW = json.load(f)

    root = "BA_SBM_SW_phi_vary"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)


#########################################################################################################
    #DELTE
    #params_BA["carbon_price_duration"] = 3
    
    #BA
    
    params_BA["carbon_price_increased_lower"] = 0
    params_list_no_tau_BA = produce_param_list_stochastic(params_BA, property_values_list, property_varied)

    params_BA["carbon_price_increased_lower"] = 0.1
    params_list_low_tau_BA = produce_param_list_stochastic(params_BA, property_values_list, property_varied)

    params_BA["carbon_price_increased_lower"] = 1
    params_list_high_tau_BA = produce_param_list_stochastic(params_BA, property_values_list, property_varied)

    params_list_BA = params_list_no_tau_BA + params_list_low_tau_BA + params_list_high_tau_BA
    
##########################################################################################################
    #DELTE
    #params_SBM["carbon_price_duration"] = 3
    
    #SBM
    params_SBM["carbon_price_increased_lower"] = 0    
    params_list_no_tau_SBM = produce_param_list_stochastic(params_SBM, property_values_list, property_varied)

    params_SBM["carbon_price_increased_lower"] = 0.1  
    params_list_low_tau_SBM = produce_param_list_stochastic(params_SBM, property_values_list, property_varied)

    params_SBM["carbon_price_increased_lower"] = 1 
    params_list_high_tau_SBM = produce_param_list_stochastic(params_SBM, property_values_list, property_varied)

    params_list_SBM = params_list_no_tau_SBM + params_list_low_tau_SBM + params_list_high_tau_SBM

##########################################################################################################
    #DELTE
    #params_SW["carbon_price_duration"] = 3

    #SW
    params_SW["carbon_price_increased_lower"] = 0    
    params_list_no_tau_SW = produce_param_list_stochastic(params_SW, property_values_list, property_varied)

    params_SW["carbon_price_increased_lower"] = 0.1   
    params_list_low_tau_SW = produce_param_list_stochastic(params_SW, property_values_list, property_varied)

    params_SW["carbon_price_increased_lower"] = 1  
    params_list_high_tau_SW = produce_param_list_stochastic(params_SW, property_values_list, property_varied)

    params_list_SW = params_list_no_tau_SW + params_list_low_tau_SW + params_list_high_tau_SW
    
    
############################################################################################################################
    #RUN THE STUFF
    params_list = params_list_BA + params_list_SBM + params_list_SW
    print("TOTAL RUNS", len(params_list))

    emissions_stock_serial = multi_emissions_stock(params_list)
    emissions_array = emissions_stock_serial.reshape(3, 3, property_reps, params_BA["seed_reps"])#2 is for BA and SBM,3 is for the 3 differents states
    emissions_array_BA = emissions_array[0]
    emissions_array_SBM = emissions_array[1]
    emissions_array_SW = emissions_array[2]

    #SAVE STUFF
    createFolder(fileName)
    
    save_object(emissions_array_BA , fileName + "/Data", "emissions_array_BA")
    save_object(emissions_array_SBM, fileName + "/Data", "emissions_array_SBM")
    save_object(emissions_array_SW, fileName + "/Data", "emissions_array_SW")
    save_object(params_BA, fileName + "/Data", "base_params_BA")
    save_object(params_SBM, fileName + "/Data", "base_params_SBM")
    save_object(params_SBM, fileName + "/Data", "base_params_SW")

    print("RUNS DONE")

###############################################################################################################
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD_BA = "package/constants/base_params_BA_phi.json",
        BASE_PARAMS_LOAD_SBM = "package/constants/base_params_SBM_phi.json",
        BASE_PARAMS_LOAD_SW = "package/constants/base_params_SW_phi.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_BA_SBM_SW_phi.json",
    )
    RUN_PLOT = 1

    if RUN_PLOT:
        phi_effect_plot.main(fileName = fileName_Figure_1)
