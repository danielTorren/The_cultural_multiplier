import numpy as np
from package.resources.utility import (
    load_object,
    save_object
)
from package.resources.run import generate_data
from scipy.interpolate import CubicSpline
from package.generating_data.multiplier_tax_sweep_gen import calc_consum
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

###########################################################################

def calc_dividend(H_m, tau_m, N):
    total_quantities_m = H_m.sum(axis = 0)
    
    tax_income_R =  np.sum(tau_m*total_quantities_m) 

    carbon_dividend =  tax_income_R/N
    return carbon_dividend

def calculate_emissions(t_max, B, N, M, a, P_L, A_matrix, sigma_matrix, nu, P_H_init,tau_max):

    #print("JHOHIO")
    a_matrix = np.tile(a, (N, 1))
    E = 0
    tau_m = np.linspace(tau_max,0,M)
    P_H = P_H_init + tau_m #ASSUMES LOWER IS 0 

    if tau_max > 0:
        carbon_dividend = 0
        # I need to do an inital step outside to match the matrix runs
        H_m = calc_consum(B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix,M)
        carbon_dividend = calc_dividend(H_m, tau_m, N)
        for i in range(t_max):
            instant_B = B + carbon_dividend
            H_m = calc_consum(instant_B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix,M)
            carbon_dividend = calc_dividend(H_m, tau_m, N)
            #print("carbon_dividend",carbon_dividend)
            E += np.sum(H_m) #np.sum no axis returns the sum of the entire matrix
    else:
        H_m = calc_consum(B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix, M)
        E = (t_max-1)*np.sum(H_m)

    return E

##################################################################################################
#REVERSE Engineer the carbon price based on the final emissions

def objective_function_to_min(tau, *args):

    emissions_target, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init = args

    E = calculate_emissions(t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init,tau)
    #print("E,tau: ",E, tau, emissions_target)
    convergence_val = (emissions_target - E )#*1e9
    
    return convergence_val

def newton_raphson_with_bounds(func, initial_guess, args_E,bounds = None, tol=1e-3, max_iter=100, delta=1e-3):
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

    # NOT SURE IM COMFORTABLE WITH THIS

    root = initial_guess
    iterations = 0
    #test_a = func(root, *args_E)
    #print(test_a)
    #quit()
    while abs(func(root, *args_E)) > tol and iterations < max_iter:
        #print("NEW TRY")
        #print("Current root", root)
        # Numerical approximation of the derivative
        derivative_approx = (func(root + delta, *args_E) - func(root - delta, *args_E)) / (2 * delta)
        #print("derivative_approx",derivative_approx)
        # Update the root using Newton-Raphson formula
        root = root - func(root, *args_E)/derivative_approx
        #print("new root", root)
        # Check if the updated root is within bounds
        if bounds is not None:
            root = max(min(root, bounds[1]), bounds[0])

        iterations += 1

    #print(" root, iterations ", root, iterations )

    return root, iterations

def optimize_tau(emissions_val, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, initial_guess,tau_lower_bound, tau_upper_bound):

    args_E = (emissions_val,t_max, B, N, M, a, P_L, A, sigma, nu,P_H_init)
    bounds = (tau_lower_bound, tau_upper_bound) #THESE ARE BOUNDS FOR THE TAX
    #print("NEXT SEED")
    root, iterations = newton_raphson_with_bounds(objective_function_to_min,initial_guess = initial_guess, args_E = args_E, bounds = bounds,tol=1e-6, max_iter=100, delta=1e-7)
    
    return root

def calc_tau_static_preference(emissions_min, emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, initial_guess_min, initial_guess_max,tau_lower_bound, tau_upper_bound):
    #TO ACHIEVE THE GREATEST EMISSIONS YOU NEED THE LOWEST TAU
    #print("ho")
    min_tau = optimize_tau(emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, initial_guess_min,tau_lower_bound, tau_upper_bound)
    print("MIN DONE")
    
    #TO ACHIEVE THE LOWEST EMISSIONS YOU NEED THE GREATEST TAU
    max_tau = optimize_tau(emissions_min, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, initial_guess_max,tau_lower_bound, tau_upper_bound)
    print("MAX DONE")

    return min_tau, max_tau

def calculate_emissions_wrapper(args):
    t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau = args
    return calculate_emissions(t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau)


def calc_required_static_carbon_tax(base_params, emissions_min, emissions_max, tau_lower_bound, tau_upper_bound, initial_guess_min_tau,initial_guess_max_tau,total_range_runs):
    #SET IT NOW AS THEN LATER CHANGE IT FOR THE REFERENCE RUN
    t_max = base_params["carbon_price_duration"] + base_params["burn_in_duration"]

    #REFERENCE CASE, helps with the base values
    base_params["set_seed"] = 5# DOESNT MATTER AS VARYIGN THE NETWORK STRUCTURE, BUT NEEDS TO BE SET
    base_params["carbon_price_increased"] = 0#this doesnt matter but needs to be set
    base_params["carbon_price_increased_lower"] = 0
    base_params["carbon_price_duration"] = 0#just make them begin
    base_params["alpha_change_state"] = "fixed_preferences"
    
    #print("base paras", base_params)
    #quit()

    #HOMOPHILY IS 0 in the case im using

    reference_run = generate_data(base_params)  # run the simulation

    B, N, M, a, P_L, A, sigma, nu, P_H_init = (
        reference_run.individual_expenditure_array,
        base_params["N"],
        base_params["M"],
        reference_run.sector_preferences,
        reference_run.prices_low_carbon_m,
        reference_run.low_carbon_preference_matrix,
        reference_run.low_carbon_substitutability_matrix,
        reference_run.sector_substitutability,
        reference_run.prices_high_carbon_m
        )

    print("RAN REFERENCE CASE")

    ###################################################################

    #THESE ARE THE BOUNDS OF WHAT I NEED TO ACHIEVE, I ASSUME THAT BECAUSE THE FUNCTION IS MONOTONICALLY DECREASING IN TAu then emissions between max and min value are interior?
    min_tau, max_tau = calc_tau_static_preference(emissions_min, emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu,P_H_init, initial_guess_min_tau, initial_guess_max_tau,tau_lower_bound, tau_upper_bound)
    
    print("min_tau, max_tau",min_tau, max_tau)
    #quit()
    #now calc the emissions over the appropriate range
    total_tau_range_static = np.linspace(min_tau, max_tau, total_range_runs)
    args_list = [(t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau) for tau in total_tau_range_static]
    print("TOTAL RUNS", total_range_runs)
    #emissions_array_static_full = [calculate_emissions(t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau) for tau in total_tau_range_static]
    emissions_array_static_full = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(calculate_emissions_wrapper)(i) for i in args_list)
    #print("emissions_array_static_full", emissions_array_static_full)

    return  total_tau_range_static, emissions_array_static_full

def calc_required_static_carbon_tax_seeds(base_params, property_values_list, emissions_networks,tau_lower_bound, tau_upper_bound, total_range_runs):
    
    emissions_min, emissions_max = np.amin(emissions_networks), np.amax(emissions_networks)
    #print("emissions_min, emissions_max", emissions_min, emissions_max)
    #quit()

    initial_guess_min_tau, initial_guess_max_tau  = min(property_values_list), max(property_values_list)
    #print("ahassa")
    tau_list, emissions_list = calc_required_static_carbon_tax(base_params, emissions_min, emissions_max, tau_lower_bound, tau_upper_bound, initial_guess_min_tau, initial_guess_max_tau, total_range_runs)
    #print("emissions_list", emissions_list)
    #print("tau_list",tau_list)
    #quit()
    return  np.asarray(tau_list), np.asarray(emissions_list)


def calc_M_vector(tau_social_vec ,tau_static_vec, emissions_social, emissions_static_full):
    #I WANT TO INTERPOLATE WHAT THE TAU VALUES OF THE STATIC ARE THAT GIVE THE SOCIAL

    #WHY DOES CUBIC SPLINES REQUIRE INCREASIGN VALUES??
    reverse_emissions_static_full = emissions_static_full[::-1]
    reverse_tau_static_vec = tau_static_vec[::-1]

    #print(emissions_static_full)
    #print(tau_static_vec)

    #plt.plot(tau_static_vec, emissions_static_full)
    #plt.show()
    #quit()
    reverse_cs_static_input_emissions_output_tau = CubicSpline(reverse_emissions_static_full, reverse_tau_static_vec)# given the emissions what is the predicted tau
    predicted_reverse_tau_static = reverse_cs_static_input_emissions_output_tau(emissions_social)

    trans_predicted_reverse_tau_static = np.transpose(predicted_reverse_tau_static,(3,0,1,2))
    
    trans_M_vals = 1 - tau_social_vec/trans_predicted_reverse_tau_static

    M_vals = np.transpose(trans_M_vals,(1,2,3,0))

    return M_vals

def main(
    fileName,
    tau_lower_bound = -0.99, 
    tau_upper_bound = 1e6,#OBSERDLY LARGE UPPER LIMIT
    total_range_runs = 10000
) -> None:
    emissions_SW = load_object(fileName + "/Data","emissions_SW")

    emissions_SBM = load_object(fileName + "/Data","emissions_SBM")
    emissions_BA = load_object(fileName + "/Data","emissions_BA")
    emissions_networks = np.asarray([emissions_SW[1:],emissions_SBM[1:],emissions_BA[1:]])# DONT INCLUDE FIXED PREFERCNCES
    #emissions_networks = np.asarray([emissions_SW,emissions_SBM,emissions_BA])

    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 

    #"""

    print("STARTING CALC")
    tau_matrix, emissions_matrix = calc_required_static_carbon_tax_seeds(base_params,property_values_list, emissions_networks,tau_lower_bound, tau_upper_bound, total_range_runs)#EMISSIONS CAN BE ANY OF THEM

    print("CALCULATED DATA")
    save_object(tau_matrix,fileName + "/Data", "tau_matrix")
    save_object(emissions_matrix,fileName + "/Data", "emissions_matrix")
    #"""
    ##############

    #tau_matrix = load_object(fileName + "/Data", "tau_matrix")
    #emissions_matrix = load_object(fileName + "/Data", "emissions_matrix")

    M_vals_networks = calc_M_vector(property_values_list , tau_matrix,emissions_networks, emissions_matrix)
    save_object(M_vals_networks,fileName + "/Data", "M_vals_networks")

if __name__ == '__main__':
    plots = main(
        fileName= "results/asym_tax_sweep_networks_13_09_20__20_05_2024"#asym_tax_sweep_networks_17_31_30__10_05_2024"
    )