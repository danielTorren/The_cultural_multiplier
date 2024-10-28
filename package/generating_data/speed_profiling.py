# imports
from package.resources.run import generate_data
import cProfile
import pstats

def main(base_params): 
    Data = generate_data(base_params)  # run the simulation
    print("Em", Data.total_carbon_emissions_stock)

if __name__ == '__main__':

    ###################################################################
    base_params = {
    "phi": 0.02,
    "network_type": "SW",
    "carbon_price_increased": 0,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    
    "alpha_change_state": "dynamic_identity_determined_weights",#"dynamic_socially_determined_weights",#"dynamic_identity_determined_weights"
    "seed_reps": 25,
    "network_structure_seed": 10, 
    "preferences_seed": 77, 
    "shuffle_homophily_seed": 2,
    "shuffle_coherance_seed": 5,
    "set_seed": 4, 
    "carbon_price_duration": 360, 
    "burn_in_duration": 0, 
    "N": 10000, 
    "M": 2, 
    "sector_substitutability": 2, 
    "low_carbon_substitutability": 2, 
     
    "a_preferences": 2, 
    "b_preferences": 2, 
    "clipping_epsilon_init_preference": 1e-5,
    "confirmation_bias": 5, 
    "init_carbon_price": 0, 
    "homophily_state": 0,
    "coherance_state": 0.95,
    "SF_nodes": 11,
    "SF_green_or_brown_hegemony": 0,
    "SBM_block_num": 2,
    "SBM_network_density_input_intra_block": 0.2,
    "SBM_network_density_input_inter_block": 0.005,
    "SW_network_density": 0.1,
    "SW_prob_rewire": 0.1
    }

    
    # Create a profiler object
    pr = cProfile.Profile()

    # Start profiling
    pr.enable()

    # Run your model code here
    main(base_params)

    # Stop profiling
    pr.disable()

    # Save profiling results to a file
    pr.dump_stats('profile_results.prof')

    # Analyze the profiling results
    p = pstats.Stats('profile_results.prof')
    p.sort_stats('cumulative').print_stats(10)

    # Visualize with snakeviz
    # Run in terminal: snakeviz profile_results.prof