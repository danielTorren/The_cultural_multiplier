import numpy as np
from package.resources.run import generate_data
from copy import deepcopy

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
    common_vector_denominator = Omega_m*P_L + P_H
    chi_pow = chi_m**nu
    Z = chi_pow*common_vector_denominator   
    return Z

def calc_consum(B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix):
    Omega_m = calc_Omega_m(P_H,P_L,A_matrix, sigma_matrix)
    n_tilde_m = calc_n_tilde_m(A_matrix,Omega_m,sigma_matrix)
    chi_m = calc_chi_m_nested_CES(a_matrix,n_tilde_m,nu, P_H)
    Z = calc_Z(Omega_m,P_L,P_H,chi_m,nu)
    H_m = (B*((chi_m**nu).T)).T/Z
    #H_m = (B*(chi_m**nu))/Z #I want this matrix to be NxM
    #print(H_m.shape)
    #quit()

    return H_m

def calc_dividend(H_m, tau, N):
    total_quantities_system = sum(H_m)
    tax_income_R =  tau*total_quantities_system    
    carbon_dividend =  tax_income_R/N
    return carbon_dividend

def calculate_emissions(t_max, B, N, M, a, P_L, A_matrix, sigma_matrix, nu, P_H_init,tau):

    a_matrix = np.tile(a, (N, 1))
    E = 0
    P_H = P_H_init + tau

    if tau > 0:
        for i in range(t_max):
            carbon_dividend = 0
            H_m = calc_consum(B + carbon_dividend, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix)
            carbon_dividend = calc_dividend(H_m, tau, N)
            E += sum(H_m) 
    else:
        H_m = calc_consum(B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix)
        E = t_max*(sum(H_m))
    return E

##################################################################################################
#REVERSE Engineer the carbon price based on the final emissions

def objective_function(tau, *args):

    emissions_target, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init = args

    E = calculate_emissions(t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init,tau)

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

    root = initial_guess
    iterations = 0

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

def optimize_PH(emissions_val, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, initial_guess,lower_bound, upper_bound):

    args_E = (emissions_val,t_max, B, N, M, a, P_L, A, sigma, nu,P_H_init)
    bounds = (lower_bound, upper_bound) #THESE ARE BOUNDS FOR THE TAX
    #print("NEXT SEED")
    root, iterations = newton_raphson_with_bounds(objective_function,initial_guess = initial_guess, args_E = args_E, bounds = bounds,tol=1e-6, max_iter=100, delta=1e-7)
    
    return root

def calc_tau_static_preference(emissions_min, emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, initial_guess_min, initial_guess_max,lower_bound, upper_bound):
    #TO ACHIEVE THE GREATEST EMISSIONS YOU NEED THE LOWEST TAU
    min_tau = optimize_PH(emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, initial_guess_min,lower_bound, upper_bound)
    #TO ACHIEVE THE LOWEST EMISSIONS YOU NEED THE GREATEST TAU
    max_tau = optimize_PH(emissions_min, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, initial_guess_max,lower_bound, upper_bound)

    return min_tau, max_tau

def calc_required_static_carbon_tax(base_params, emissions_min, emissions_max, set_seed, lower_bound, upper_bound, initial_guess_min_tau,initial_guess_max_tau,total_range_runs):
    #SET IT NOW AS THEN LATER CHANGE IT FOR THE REFERENCE RUN
    t_max = base_params["carbon_price_duration"] + base_params["burn_in_duration"]

    #REFERENCE CASE, helps with the base values
    base_params["set_seed"] = set_seed
    base_params["carbon_price_increased"] = 0#this doesnt matter but needs to be set
    base_params["carbon_price_increased_lower"] = 0
    base_params["carbon_price_duration"] = 0#just make them begin
    base_params["alpha_change_state"] = "fixed_preferences"

    reference_run = generate_data(base_params)  # run the simulation

    B, N, M, a, P_L, A, sigma, nu, P_H_init = (
        reference_run.individual_expenditure_array,
        base_params["N"],
        base_params["M"],
        reference_run.sector_preferences,
        reference_run.prices_low_carbon,
        reference_run.low_carbon_preference_matrix_init,
        np.asarray(reference_run.low_carbon_substitutability_array_list),
        reference_run.sector_substitutability,
        reference_run.prices_high_carbon
        )
    
    ###################################################################

    #THESE ARE THE BOUNDS OF WHAT I NEED TO ACHIEVE, I ASSUME THAT BECAUSE THE FUNCTION IS MONOTONICALLY DECREASING IN TAu then emissions between max and min value are interior?
    #initial_guess_min_tau = min(property_values_list)
    #initial_guess_max_tau = max(property_values_list)
    #print("INTIAL GUESSES", initial_guess_min_tau, initial_guess_max_tau)

    min_tau, max_tau = calc_tau_static_preference(emissions_min, emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu,P_H_init, initial_guess_min_tau, initial_guess_max_tau,lower_bound, upper_bound)
    print("OPTIMISATION DONE, SEED = ", set_seed)
    #now calc the emissions over the appropriate range
    total_tau_range_static = np.linspace(min_tau, max_tau, total_range_runs)
    emissions_array_static_full = [calculate_emissions(t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau) for tau in total_tau_range_static]
    print("CALCULATED FULL EMISSIONS, SEED = ", set_seed)

    return  total_tau_range_static, emissions_array_static_full

def calc_required_static_carbon_tax_seeds(base_params, property_values_list, emissions_networks,lower_bound, upper_bound, total_range_runs):

    emissions_seeds = np.transpose(emissions_networks,(3,0,1,2))# shape : from netowrk, scenario, reps , seeds to SHAPE : seeds, netowrk, scenario, reps

    tau_matrix = []
    emissions_matrix = []
    
    initial_guess_min_tau = min(property_values_list)
    initial_guess_max_tau = max(property_values_list)

    for v in range(base_params["seed_reps"]):#
        set_seed = int(v+1)
        emissions_seed_runs = emissions_seeds[v]

        emissions_min, emissions_max = np.amin(emissions_seed_runs), np.amax(emissions_seed_runs)
        #print("emissions_min, emissions_max ",emissions_min, emissions_max )
        #pass
        base_params_copy = deepcopy(base_params)

        tau_list, emissions_list = calc_required_static_carbon_tax(base_params_copy, emissions_min, emissions_max, set_seed, lower_bound, upper_bound,  initial_guess_min_tau, initial_guess_max_tau, total_range_runs)
        
        initial_guess_min_tau = min(tau_list)#UPDATE THE STARTING VALUES
        initial_guess_max_tau = max(tau_list)

        #print("min max",initial_guess_min_tau, initial_guess_max_tau)

        tau_matrix.append(tau_list)
        emissions_matrix.append(emissions_list)

    
    data_tau_full = np.asarray(tau_matrix)
    data_tau = np.squeeze(data_tau_full)
    data_emissions_full = np.asarray(emissions_matrix)
    #print(data_tau.shape, data_emissions_full.shape)
    data_emissions = np.squeeze(data_emissions_full)#get rid of extra 1 dimension that doenstn do anything

    return  data_tau, data_emissions#THESE HAVE SHAPE SEED by REPS set(1000)