import numpy as np
from package.resources.utility import (
    load_object,
    save_object
)
from package.resources.run import generate_data
from scipy.interpolate import CubicSpline
from package.generating_data.preference_complete_multiplier_tax_sweep_gen import calc_consum, calc_M_vector_seeds
from joblib import Parallel, delayed
import multiprocessing
from scipy.optimize import brentq

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
    print(min_tau, max_tau)
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


def main(
    fileName,
    tau_lower_bound = -0.99, 
    tau_upper_bound = 1e6,#OBSERDLY LARGE UPPER LIMIT
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

    M_vals_networks = calc_M_vector_seeds(property_values_list , tau_matrix,emissions_networks, emissions_matrix,base_params)
    save_object(M_vals_networks,fileName + "/Data", "M_vals_networks")

if __name__ == '__main__':
    plots = main(
        fileName= "results/asym_tax_sweep_networks_13_09_20__20_05_2024",#asym_tax_sweep_networks_17_31_30__10_05_2024",
        LOAD = True
    )