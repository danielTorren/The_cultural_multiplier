import numpy as np
from package.resources.utility import (
    load_object,
    save_object
)
from package.resources.run import generate_data
from scipy.interpolate import CubicSpline
import numpy.typing as npt
from joblib import Parallel, delayed
import multiprocessing

###########################################################################
def calc_Omega_m(P_H,P_L,A_matrix, sigma_matrix):
    term_1 = (P_H*A_matrix)
    term_2 = (P_L*(1- A_matrix))
    omega_vector = (term_1/term_2)**(sigma_matrix)
    return omega_vector

def calc_n_tilde_m(A_matrix,Omega_m,sigma_matrix):
    n_tilde_m = (A_matrix*(Omega_m**((sigma_matrix-1)/sigma_matrix))+(1-A_matrix))**(sigma_matrix/(sigma_matrix-1))
    return n_tilde_m
    
def calc_chi_m_nested_CES(a,n_tilde_m,nu, P_H):
    chi_m = (a*(n_tilde_m**((nu-1)/nu)))/P_H
    return chi_m

def calc_Z(Omega_m,P_L,P_H,chi_m,nu):
    common_vector = Omega_m*P_L + P_H
    chi_pow = chi_m**nu
    no_sum_Z_terms = chi_pow*common_vector
    Z = no_sum_Z_terms.sum(axis = 1)
    return Z

def calc_consum(instant_B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix,M):
    Omega_m = calc_Omega_m(P_H,P_L,A_matrix, sigma_matrix)
    n_tilde_m = calc_n_tilde_m(A_matrix,Omega_m,sigma_matrix)
    chi_m = calc_chi_m_nested_CES(a_matrix,n_tilde_m,nu, P_H)
    Z = calc_Z(Omega_m,P_L,P_H,chi_m,nu)

    term_1 = instant_B/Z
    term_1_matrix = np.tile(term_1, (M,1)).T
    H_m = term_1_matrix*(chi_m**nu)

    return H_m

def calc_dividend(H_m, tau, N):
    total_quantities_system = np.sum(H_m)
    tax_income_R =  tau*total_quantities_system    
    carbon_dividend =  tax_income_R/N
    return carbon_dividend

def calculate_emissions(t_max, B, N, M, a, P_L, A_matrix, sigma_matrix, nu, P_H_init,tau):

    a_matrix = np.tile(a, (N, 1))
    E = 0
    P_H = P_H_init + tau

    if tau > 0:
        carbon_dividend = 0
        # I need to do an inital step outside to match the matrix runs
        H_m = calc_consum(B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix,M)
        carbon_dividend = calc_dividend(H_m, tau, N)
        for i in range(t_max):
            instant_B = B + carbon_dividend
            H_m = calc_consum(instant_B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix,M)
            carbon_dividend = calc_dividend(H_m, tau, N)
            E += np.sum(H_m) #np.sum no axis returns the sum of the entire matrix
    else:
        H_m = calc_consum(B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix, M)
        E = (t_max-1)*np.sum(H_m)

    return E

##################################################################################################
#REVERSE Engineer the carbon price based on the final emissions

def calculate_emissions(base_params):
    reference_run = generate_data(base_params)
    E = reference_run.total_carbon_emissions_stock
    return E

def objective_function_to_min_full_run(tau, base_params):
    
    base_params["carbon_price_increased"] = tau

    E = calculate_emissions(base_params)
    convergence_val = (base_params["emissions_target"] - E )#*1e9
    return convergence_val

def newton_raphson_with_bounds(func, initial_guess, base_params, bounds = None, tol=1e-3, max_iter=100, delta=1e-3):
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
    while abs(func(root, base_params)) > tol and iterations < max_iter:
        # Numerical approximation of the derivative
        derivative_approx = (func(root + delta, base_params) - func(root - delta, base_params)) / (2 * delta)
        # Update the root using Newton-Raphson formula
        root = root - func(root, base_params)/derivative_approx
        # Check if the updated root is within bounds
        if bounds is not None:
            root = max(min(root, bounds[1]), bounds[0])

        iterations += 1

    return root, iterations

def optimize_tau(base_params,initial_guess,tau_lower_bound, tau_upper_bound):

    bounds = (tau_lower_bound, tau_upper_bound) #THESE ARE BOUNDS FOR THE TAX
    root, iterations = newton_raphson_with_bounds(objective_function_to_min_full_run, initial_guess = initial_guess, base_params = base_params, bounds = bounds,tol=1e-6, max_iter=200, delta=1e-5)
    return root

def calc_tau_static_preference(emissions_min, emissions_max, base_params, initial_guess_min, initial_guess_max,tau_lower_bound, tau_upper_bound):
    #TO ACHIEVE THE GREATEST EMISSIONS YOU NEED THE LOWEST TAU
    base_params["emissions_target"] = emissions_max
    min_tau = optimize_tau(base_params, initial_guess_min,tau_lower_bound, tau_upper_bound)

    base_params["emissions_target"] = emissions_min
    #TO ACHIEVE THE LOWEST EMISSIONS YOU NEED THE GREATEST TAU
    max_tau = optimize_tau(base_params, initial_guess_max,tau_lower_bound, tau_upper_bound)

    return min_tau, max_tau

def calc_required_static_carbon_tax(base_params, emissions_min, emissions_max, tau_lower_bound, tau_upper_bound, initial_guess_min_tau,initial_guess_max_tau,total_range_runs):
    #SET IT NOW AS THEN LATER CHANGE IT FOR THE REFERENCE RUN
    #REFERENCE CASE, helps with the base values
    base_params["carbon_price_increased"] = 0#this doesnt matter but needs to be set
    base_params["carbon_price_increased"] = 0#this doesnt matter but needs to be set

    base_params["alpha_change_state"] = "fixed_preferences"#THIS IS THE IMPORTANT ONE!

    min_tau, max_tau = calc_tau_static_preference(emissions_min, emissions_max, base_params,initial_guess_min_tau, initial_guess_max_tau,tau_lower_bound, tau_upper_bound)
    #now calc the emissions over the appropriate range
    total_tau_range_static = np.linspace(min_tau, max_tau, total_range_runs)

    emissions_array_static_full = []
    for tau in total_tau_range_static:
        base_params["carbon_price_increased"] = tau
        E = calculate_emissions(base_params)
        emissions_array_static_full.append(E)

    return  total_tau_range_static, emissions_array_static_full

def calc_required_static_carbon_tax_seeds(seeds_data, base_params, property_values_list, emissions_networks_seed,tau_lower_bound, tau_upper_bound, total_range_runs):
    base_params.update(seeds_data)
    emissions_min, emissions_max = np.amin(emissions_networks_seed), np.amax(emissions_networks_seed)
    initial_guess_min_tau, initial_guess_max_tau  = min(property_values_list), max(property_values_list)

    tau_list, emissions_list = calc_required_static_carbon_tax(base_params, emissions_min, emissions_max, tau_lower_bound, tau_upper_bound, initial_guess_min_tau, initial_guess_max_tau, total_range_runs)

    return  np.asarray(tau_list), np.asarray(emissions_list)

def calc_required_static_carbon_tax_multi_seeds(
        seeds_data_list: list[dict], base_params, property_values_list, emissions_networks,tau_lower_bound, tau_upper_bound, total_range_runs
) -> npt.NDArray:
    
    emissions_networks_trans = np.transpose(emissions_networks, (3,0,1,2))
    seeds_em_data_list = zip(seeds_data_list,emissions_networks_trans )
    num_cores = multiprocessing.cpu_count()

    res= Parallel(n_jobs=num_cores, verbose=10)(
        delayed(calc_required_static_carbon_tax_seeds)(seeds, base_params, property_values_list, emissions_networks_seed,tau_lower_bound, tau_upper_bound, total_range_runs) for seeds, emissions_networks_seed in seeds_em_data_list
    )
    tau_list, emissions_list = zip(
        *res
    )
    return tau_list, emissions_list

def calc_predicted_reverse_tau_static(tau_static_vec, emissions_social, emissions_static_full):

    reverse_emissions_static_full = emissions_static_full[::-1]
    reverse_tau_static_vec = tau_static_vec[::-1]
    reverse_cs_static_input_emissions_output_tau = CubicSpline(reverse_emissions_static_full, reverse_tau_static_vec)# given the emissions what is the predicted tau
    predicted_tau_static = reverse_cs_static_input_emissions_output_tau(emissions_social)

    return predicted_tau_static

def calc_predicted_reverse_tau_static_seeds(tau_static_list, emissions_social, emissions_static_list):

    emissions_social_trans = np.transpose(emissions_social,(3,0,1,2))#move seed to front

    e_tau_list = zip(emissions_social_trans, tau_static_list, emissions_static_list)

    predicted_reverse_tau_static = [calc_predicted_reverse_tau_static(tau_vec, emissions_social_seed, emissions_vec) for emissions_social_seed, tau_vec, emissions_vec in e_tau_list]

    return  np.asarray(predicted_reverse_tau_static)

def calc_M_vector_seeds(tau_social_vec ,predicted_reverse_tau_static_arr):

    trans_M_vals = 1 - tau_social_vec/predicted_reverse_tau_static_arr

    M_vals = np.transpose(trans_M_vals,(1,2,3,0))#move seed to back

    return M_vals

def calc_ratio_seeds(tau_social_vec,predicted_reverse_tau_static_arr):

    trans_ratio_vals = tau_social_vec/predicted_reverse_tau_static_arr

    ratio_vals = np.transpose(trans_ratio_vals,(1,2,3,0))#move seed to back

    return ratio_vals


def reconstruct_seeds_list(params: dict) -> list[dict]:
    seeds_labels = ["preferences_seed", "network_structure_seed", "shuffle_homophily_seed", "shuffle_coherance_seed"]
    list_dicts = []
    for j in range(params["seed_reps"]):
        seeds_dict = {}
        for k, label in enumerate(seeds_labels):
            seeds_dict[label] = int(10*k + j + 1)
        list_dicts.append(
            seeds_dict
        )  
    return list_dicts

def main(
    fileName,
    RUN = 0,
    tau_lower_bound = -0.6, 
    tau_upper_bound = 10,
    total_range_runs = 100
) -> None:
    
    emissions_SW = load_object(fileName + "/Data","emissions_SW")
    emissions_SBM = load_object(fileName + "/Data","emissions_SBM")
    emissions_SF = load_object(fileName + "/Data","emissions_SF")

    emissions_networks = np.asarray([emissions_SW[1:],emissions_SBM[1:],emissions_SF[1:]])# DONT INCLUDE FIXED PREFERCNCES
    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 
    
    

    if RUN: 
        #"""
        #################################################################
        #Recover the seeds used
        seeds_data_dicts = reconstruct_seeds_list(base_params)
        print("TOTAL SEEDS: ", len(seeds_data_dicts))
        #################################################################  

        print("STARTING CALC")
        tau_list_matrix, emissions_list_matrix = calc_required_static_carbon_tax_multi_seeds(seeds_data_dicts, base_params,property_values_list, emissions_networks,tau_lower_bound, tau_upper_bound, total_range_runs)#EMISSIONS CAN BE ANY OF THEM

        print("CALCULATED DATA")
        save_object(tau_list_matrix,fileName + "/Data", "tau_list_matrix")
        save_object(emissions_list_matrix,fileName + "/Data", "emissions_list_matrix")
        #"""

        ##########################################
    else:
        tau_list_matrix = load_object(fileName + "/Data", "tau_list_matrix")
        emissions_list_matrix = load_object(fileName + "/Data", "emissions_list_matrix")

    predicted_reverse_tau_static = calc_predicted_reverse_tau_static_seeds( tau_list_matrix, emissions_networks, emissions_list_matrix)
    list_M_networks = calc_M_vector_seeds(property_values_list , predicted_reverse_tau_static)
    list_ratio_networks = calc_ratio_seeds(property_values_list , predicted_reverse_tau_static)
    save_object(list_M_networks,fileName + "/Data", "list_M_networks")

    tau_static = np.transpose(predicted_reverse_tau_static,(1,2,3,0))#move seed to back
    save_object(tau_static,fileName + "/Data", "tau_static")
    save_object(list_ratio_networks,fileName + "/Data", "list_ratio_networks")

if __name__ == '__main__':
    plots = main(
        fileName= "results/tax_sweep_networks_18_23_13__19_08_2024",
        RUN = 0
    )