"""Runs a single simulation to produce data which is saved

A module that use dictionary of data for the simulation run. The single shot simulztion is run
for a given initial set seed.



Created: 10/10/2022
"""
# imports
from package.model.network_matrix import Network_Matrix
from joblib import Parallel, delayed
import multiprocessing
import numpy as np

def main(
    base_params
) -> str: 

    num_cores = multiprocessing.cpu_count()
    seed_reps = 1
    params_list = []
    for v in range(seed_reps):
            base_params["set_seed"] = int(v+1)
            params_list.append(base_params.copy())  
    #print("SETUP DONE")
    """
    a = generate_data_alt(params_list[0])
    c = np.asarray([n.low_carbon_preferences for n in a.agent_list])
    #e = a.low_carbon_preference_matrix_init
    #print("a", c)

    #print("ORIGINAL DONE!")
    
    b = generate_data_alt_matrix(params_list[0])
    d = b.low_carbon_preference_matrix
    #f = b.low_carbon_preference_matrix_init
    
    #print("b",d )
    print("diff", c-d)


    quit()
    """
    data_parallel_matrix = [generate_data_alt_matrix(i) for i in params_list]
    #quit()
    #data_parallel_matrix  = Parallel(n_jobs=num_cores, verbose=10)(
    #    delayed(generate_data_alt_matrix)(i) for i in params_list
    #)

    print("MATRIX DONE!")

    data_parallel = [generate_data_alt(i) for i in params_list]
    #data_parallel = Parallel(n_jobs=num_cores, verbose=10)(
    #    delayed(generate_data_alt)(i) for i in params_list
    #)
    print("ORIGINAL DONE!")

    
    #"""
    emissions = np.asarray([data.total_carbon_emissions_stock for data in data_parallel])
    emissions_matrix =  np.asarray([data.total_carbon_emissions_stock for data in data_parallel_matrix])

    """
    H = np.asarray([[x.H_m for x in data.agent_list] for data in data_parallel])
    H_matrix = np.asarray([data.H_m_matrix for data in data_parallel_matrix])

    Omega = np.asarray([[x.Omega_m for x in data.agent_list] for data in data_parallel])
    Omega_matrix = np.asarray([data.Omega_m_matrix for data in data_parallel_matrix])
    #print("diff Omega", Omega_matrix - Omega)

    n_tilde_m = np.asarray([[x.n_tilde_m for x in data.agent_list] for data in data_parallel])
    n_tilde_m_matrix = np.asarray([data.n_tilde_m_matrix for data in data_parallel_matrix])
    #print("diff n_tilde_m", n_tilde_m - n_tilde_m_matrix)

    chi = np.asarray([[x.chi_m for x in data.agent_list] for data in data_parallel])
    chi_matrix = np.asarray([data.chi_m_tensor for data in data_parallel_matrix])
    #print("diff chi",chi - chi_matrix)

    Z = np.asarray([[x.Z for x in data.agent_list] for data in data_parallel])
    Z_matrix = np.asarray([data.Z_vec for data in data_parallel_matrix])
    
    #print("diff Z", Z - Z_matrix)

    preferences = np.asarray([[x.low_carbon_preferences for x in data.agent_list] for data in data_parallel])
    preferences_matrix = np.asarray([data.low_carbon_preference_matrix for data in data_parallel_matrix])
    """
    
    #print(" emissions", emissions)
    #print("emissions_matrix", emissions_matrix)
    print(" percent diff emissions", (emissions_matrix/emissions)*100-100)
    #print("diff H", H_matrix - H)
    #print("diff preferences", preferences - preferences_matrix)

def generate_data_alt(parameters: dict):

    parameters["time_steps_max"] = parameters["burn_in_duration"] + parameters["carbon_price_duration"]

    #print("tim step max", parameters["time_steps_max"],parameters["burn_in_duration"], parameters["carbon_price_duration"])
    social_network = Network(parameters)

    #### RUN TIME STEPS
    while social_network.t < parameters["time_steps_max"]:
        social_network.next_step()
        #print("step", social_network.t)

    #print("DONE")
    return social_network

def generate_data_alt_matrix(parameters: dict):

    parameters["time_steps_max"] = parameters["burn_in_duration"] + parameters["carbon_price_duration"]
    #print(parameters["time_steps_max"])
    social_network = Network_Matrix(parameters)

    #### RUN TIME STEPS
    while social_network.t < parameters["time_steps_max"]:
        social_network.next_step()
        #print("step", social_network.t)

    #print("DONE")
    return social_network

if __name__ == '__main__':
    
    base_params = {
    "imitation_state": "common_knowledge",
    "alpha_change_state": "dynamic_identity_determined_weights",#"dynamic_socially_determined_weights",#"dynamic_identity_determined_weights"
    "network_type": "SBM",
    "save_timeseries_data_state": 0,
    "compression_factor_state": 1,
    "heterogenous_intrasector_preferences_state": 1,
    "heterogenous_carbon_price_state": 0,
    "heterogenous_sector_substitutabilities_state": 1,
    "SBM_block_heterogenous_individuals_substitutabilities_state": 1,
    "heterogenous_phi_state": 0,
    "imperfect_learning_state": 1,
    "vary_seed_state": "network",

    "set_seed": 5,
    "network_structure_seed": 8, 
    "preferences_seed": 14,
    "shuffle_homophily_seed":20,
	"learning_seed":10, 
    "carbon_price_duration": 1000, 
    "burn_in_duration": 0, 
    "N": 20, 
    "M": 2, 
    "sector_substitutability": 2, 
    "low_carbon_substitutability_lower": 2, 
    "low_carbon_substitutability_upper": 5, 
    "a_identity": 2, 
    "b_identity": 2, 
    "clipping_epsilon": 0, 
    "clipping_epsilon_init_preference": 1e-5,
    "std_low_carbon_preference": 0.01, 

    "confirmation_bias": 1, 
    
    "init_carbon_price": 0, 
    "phi_lower": 0.02, 
    "homophily_state": 0.1,
    "SW_network_density": 0.1,
    "SW_prob_rewire": 0.5,
    "BA_nodes": 11,
    "BA_green_or_brown_hegemony": 0,
    "SBM_block_num": 2,
    "SBM_network_density_input_intra_block": 0.6,#0.2,
    "SBM_network_density_input_inter_block": 0.3,#0.005,
    "SBM_sub_add_on": 5,
    "carbon_price_increased_lower": 0.2
    }
    
    main(base_params=base_params)
