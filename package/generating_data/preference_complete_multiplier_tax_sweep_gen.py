import numpy as np
import matplotlib.pyplot as plt
from seaborn import axes_style
from package.resources.utility import (
    load_object,
    save_object
)
from package.resources.run import generate_data
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed
import multiprocessing
from scipy.optimize import brentq

###########################################################################
def calc_Omega_m(P_H,P_L,A_matrix, sigma_matrix):
    #CORRECT I THINK
    term_1 = (P_H*A_matrix)
    term_2 = (P_L*(1- A_matrix))
    omega_vector = (term_1/term_2)**(sigma_matrix)
    return omega_vector

def calc_n_tilde_m(A_matrix,Omega_m,sigma_matrix):
    #LASO CORRECT
    n_tilde_m = (A_matrix*(Omega_m**((sigma_matrix-1)/sigma_matrix))+(1-A_matrix))**(sigma_matrix/(sigma_matrix-1))
    return n_tilde_m
    
def calc_chi_m_nested_CES(a,n_tilde_m,nu, P_H):
    #CORRECT
    chi_m = (a*(n_tilde_m**((nu-1)/nu)))/P_H
    return chi_m

def calc_Z(Omega_m,P_L,P_H,chi_m,nu):
    common_vector = Omega_m*P_L + P_H
    chi_pow = chi_m**nu
    no_sum_Z_terms = chi_pow*common_vector
    Z = no_sum_Z_terms.sum(axis = 1)#SO NOW ITS 1d vector for each individual

    #common_vector_denominator = Omega_m*P_L + P_H
    #chi_pow = chi_m**nu
    #Z = chi_pow*common_vector_denominator  
    #print("no_sum_Z_terms", no_sum_Z_terms)
    #print("Z", Z) 
    #quit()
    return Z

def calc_consum(instant_B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix,M):
    Omega_m = calc_Omega_m(P_H,P_L,A_matrix, sigma_matrix)
    n_tilde_m = calc_n_tilde_m(A_matrix,Omega_m,sigma_matrix)
    chi_m = calc_chi_m_nested_CES(a_matrix,n_tilde_m,nu, P_H)
    Z = calc_Z(Omega_m,P_L,P_H,chi_m,nu)

    term_1 = instant_B/Z
    term_1_matrix = np.tile(term_1, (M,1)).T
    H_m = term_1_matrix*(chi_m**nu)# I WROTE IT OUT THE SAME WAY AS IN THE NETWORK MATRIX

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
            #print("carbon_dividend",carbon_dividend)
            E += np.sum(H_m) #np.sum no axis returns the sum of the entire matrix
    else:
        H_m = calc_consum(B, P_L, A_matrix, sigma_matrix, nu, P_H, a_matrix, M)       
        E = (t_max-1)*np.sum(H_m)
        #print("H_m", H_m, np.sum(H_m))
        #quit()

    return E

##################################################################################################
#REVERSE Engineer the carbon price based on the final emissions

def objective_function_to_min(tau, *args):

    emissions_target, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init = args
    #print("t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init,tau", t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init,tau)
    E = calculate_emissions(t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init,tau)
    #print("E,tau: ",E, tau, emissions_target)
    convergence_val = (emissions_target - E )#*1e9
    #print("convergence_val", convergence_val)
    
    return convergence_val


def optimize_tau(emissions_val, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau_lower_bound, tau_upper_bound):

    args_E = (emissions_val,t_max, B, N, M, a, P_L, A, sigma, nu,P_H_init)
    #bounds = (tau_lower_bound, tau_upper_bound) #THESE ARE BOUNDS FOR THE TAX
    
    #print("conditions",emissions_val, initial_guess,tau_lower_bound, tau_upper_bound)
    root = brentq(f = objective_function_to_min, a = tau_lower_bound, b = tau_upper_bound, args=args_E, maxiter= 100)
    #print("root", root)


    return root

def calc_tau_static_preference(emissions_min, emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau_lower_bound, tau_upper_bound):
    
    #TO ACHIEVE THE GREATEST EMISSIONS YOU NEED THE LOWEST TAU
    min_tau = optimize_tau(emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau_lower_bound, tau_upper_bound)
    #print("MIN DONE")
    
    #TO ACHIEVE THE LOWEST EMISSIONS YOU NEED THE GREATEST TAU
    max_tau = optimize_tau(emissions_min, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init,tau_lower_bound, tau_upper_bound)
    #print("MAX DONE")

    return min_tau, max_tau

def calculate_emissions_wrapper(args):
    t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau = args
    return calculate_emissions(t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau)

def calc_required_static_carbon_tax(base_params, emissions_min, emissions_max, tau_lower_bound, tau_upper_bound,total_range_runs):
    #SET IT NOW AS THEN LATER CHANGE IT FOR THE REFERENCE RUN
    carbon_price_duration = base_params["carbon_price_duration"]
    t_max = base_params["carbon_price_duration"] + base_params["burn_in_duration"]#we record this for later when we have to run the full simulation
    ##############################################################################################################################################################
    
    #REFERENCE CASE, helps with the base values
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
        reference_run.prices_low_carbon_m,
        reference_run.low_carbon_preference_matrix,
        reference_run.low_carbon_substitutability_matrix,
        reference_run.sector_substitutability,
        reference_run.prices_high_carbon_m
        )
    
    base_params["carbon_price_duration"] = carbon_price_duration#RESET IT 

    ##################################################################################################################################################

    #THESE ARE THE BOUNDS OF WHAT I NEED TO ACHIEVE, I ASSUME THAT BECAUSE THE FUNCTION IS MONOTONICALLY DECREASING IN TAu then emissions between max and min value are interior?

    min_tau, max_tau = calc_tau_static_preference(emissions_min, emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau_lower_bound, tau_upper_bound)
    #now calc the emissions over the appropriate range
    total_tau_range_static = np.linspace(min_tau, max_tau, total_range_runs)
    emissions_array_static_full = [calculate_emissions(t_max, B, N, M, a, P_L, A, sigma, nu, P_H_init, tau) for tau in total_tau_range_static]

    return  total_tau_range_static, emissions_array_static_full

def process_seed(v, base_params, seed_emissions_networks, tau_lower_bound, tau_upper_bound, total_range_runs):
    #print("v", v)
    set_seed = int(v + 1)
    base_params["set_seed"] = set_seed

    emissions_min = np.amin(seed_emissions_networks[v])
    emissions_max = np.amax(seed_emissions_networks[v])
    #print(emissions_min, emissions_max)
    
    tau_list, emissions_list = calc_required_static_carbon_tax(base_params, emissions_min, emissions_max, tau_lower_bound, tau_upper_bound, total_range_runs)
    
    return tau_list, emissions_list
    
def calc_required_static_carbon_tax_seeds(base_params, emissions_networks,tau_lower_bound, tau_upper_bound, total_range_runs):
    
    seed_emissions_networks = np.transpose(emissions_networks, (3,0,1,2))

    results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(process_seed)(v, base_params, seed_emissions_networks, tau_lower_bound, tau_upper_bound, total_range_runs) for v in range(base_params["seed_reps"]))

    tau_data, emissions_data = zip(*results)

    return  np.asarray(tau_data), np.asarray(emissions_data)

def calc_M_vector(tau_social_vec ,tau_static_vec, emissions_social, emissions_static_full):
    #I WANT TO INTERPOLATE WHAT THE TAU VALUES OF THE STATIC ARE THAT GIVE THE SOCIAL
    reverse_emissions_static_full = emissions_static_full[::-1]
    reverse_tau_static_vec = tau_static_vec[::-1]

    reverse_cs_static_input_emissions_output_tau = CubicSpline(reverse_emissions_static_full, reverse_tau_static_vec)# given the emissions what is the predicted tau
    
    predicted_reverse_tau_static = reverse_cs_static_input_emissions_output_tau(emissions_social)

    M_vals = 1 - tau_social_vec/predicted_reverse_tau_static

    return M_vals

def process_M_seed(v, tau_social_vec ,tau_static_vec, seed_emissions_networks, emissions_static_full):

    M_vals = calc_M_vector(tau_social_vec ,tau_static_vec[v], seed_emissions_networks[v], emissions_static_full[v])
    return M_vals

def calc_M_vector_seeds(tau_social_vec ,tau_static_vec, emissions_social, emissions_static_full, base_params):


    seed_emissions_networks = np.transpose(emissions_social, (3,0,1,2))

    
    M_vals_data = []
    for v in range(base_params["seed_reps"]):
        M_vals = calc_M_vector(tau_social_vec ,tau_static_vec[v], seed_emissions_networks[v], emissions_static_full[v])
        M_vals_data.append(M_vals)

    #M_vals_data = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=0)(delayed(process_M_seed)(v, tau_social_vec ,tau_static_vec, seed_emissions_networks, emissions_static_full) for v in range(base_params["seed_reps"]))
    
    M_vals_data_arr_seeds = np.asarray(M_vals_data)

    M_vals_data_arr = np.transpose(M_vals_data_arr_seeds, (1,2,3,0))

    return M_vals_data_arr

def main(
    fileName,
    tau_lower_bound = -0.9, 
    tau_upper_bound = 1000,
    total_range_runs = 100, 
    LOAD = False
) -> None:
    emissions_SW = load_object(fileName + "/Data","emissions_SW")

    emissions_SBM = load_object(fileName + "/Data","emissions_SBM")
    emissions_BA = load_object(fileName + "/Data","emissions_BA")
    emissions_networks = np.asarray([emissions_SW[1:],emissions_SBM[1:],emissions_BA[1:]])# DONT INCLUDE FIXED PREFERCNCES
    #emissions_networks = np.asarray([emissions_SW,emissions_SBM,emissions_BA])

    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 
    print("base_params", base_params["vary_seed_state"])
    if LOAD:
        tau_matrix = load_object(fileName + "/Data", "tau_matrix")
        emissions_matrix = load_object(fileName + "/Data", "emissions_matrix")
    else:
        print("STARTING CALC")
        tau_matrix, emissions_matrix = calc_required_static_carbon_tax_seeds(base_params, emissions_networks,tau_lower_bound, tau_upper_bound, total_range_runs)#EMISSIONS CAN BE ANY OF THEM

        print("CALCULATED DATA")
        save_object(tau_matrix,fileName + "/Data", "tau_matrix")
        save_object(emissions_matrix,fileName + "/Data", "emissions_matrix")
    #"""

    M_vals_networks = calc_M_vector_seeds(property_values_list , tau_matrix, emissions_networks, emissions_matrix,base_params)
    save_object(M_vals_networks,fileName + "/Data", "M_vals_networks")

if __name__ == '__main__':
    plots = main(
        fileName= "results/tax_sweep_networks_14_58_29__17_05_2024",
        LOAD = True
    )