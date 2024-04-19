# imports
from package.resources.run import generate_data
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)
from package.plotting_data import timer_series_anti_heg_plot
import pyperclip

def main(
    base_params,
    tau_vals
) -> str: 

    root = "time_brown_heg"
    fileName = produce_name_datetime(root)
    pyperclip.copy(fileName)
    print("fileName:", fileName)
    
    base_params["carbon_price_increased_lower"] = tau_vals[0]#s0.15   
    Data_low = generate_data(base_params)  # run the simulation
    
    base_params["carbon_price_increased_lower"] = tau_vals[0]#0.2
    Data_high = generate_data(base_params)  # run the simulation

    Data_list = [Data_low,Data_high]
    createFolder(fileName)
    save_object(Data_list, fileName + "/Data", "social_networks")
    save_object(tau_vals, fileName + "/Data", "tau_vals")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':
    """
    base_params = {
    "BA_green_or_brown_hegemony": -1,
    "homophily_state": 1,
    "alpha_change_state": "dynamic_identity_determined_weights",
    "network_type": "BA",
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "heterogenous_intrasector_preferences_state": 1,
    "heterogenous_carbon_price_state": 0,
    "heterogenous_sector_substitutabilities_state": 0,
    "SBM_block_heterogenous_individuals_substitutabilities_state": 0,
    "heterogenous_phi_state": 0,
    "imperfect_learning_state": 0,
    "imitation_state": "consumption",
    "vary_seed_state": "preferences",

    "set_seed": 5,
    "network_structure_seed": 8, 
    "preferences_seed": 14,
"shuffle_seed":20,
	"learning_seed":10, 
    "carbon_price_duration": 1000, 
    "burn_in_duration": 0, 
    "N": 200, 
    "M": 3, 
    "sector_substitutability": 1.5, 
    #"low_carbon_substitutability_lower": 1.5, 
    "low_carbon_substitutability_upper": 1.5, 
    "a_identity": 2, 
    "b_identity": 2, 
    "clipping_epsilon": 0, 
    "clipping_epsilon_init_preference": 1e-5,
    "std_low_carbon_preference": 0.01, 

    "confirmation_bias": 10, 
    
    "init_carbon_price": 0, 
    "phi_lower": 0.02, 
    "BA_nodes": 11
    }
    """

    base_params = {
    "BA_green_or_brown_hegemony": -1,
    "homophily_state": 1,
    "save_timeseries_data_state": 1,
    "compression_factor_state": 1,
    "network_type": "BA",
    "heterogenous_intrasector_preferences_state": 1,
    "heterogenous_carbon_price_state": 0,
    "heterogenous_sector_substitutabilities_state": 0,
    "SBM_block_heterogenous_individuals_substitutabilities_state": 0,
    "heterogenous_phi_state": 0,
    "imperfect_learning_state": 1,
    "imitation_state": "consumption",
    "alpha_change_state": "dynamic_identity_determined_weights",
    "vary_seed_state": "learning",

    "seed_reps": 5,
    "network_structure_seed": 8, 
    "preferences_seed": 14,
"shuffle_seed":20,
	"learning_seed":10, 
    "set_seed": 4, 
    "carbon_price_duration": 1000, 
    "burn_in_duration": 0, 
    "N": 200, 
    "M": 1, 
    "sector_substitutability": 1.5, 
    "low_carbon_substitutability_lower": 1.5, 
    "low_carbon_substitutability_upper": 1.5, 
    "a_identity": 2, 
    "b_identity": 2, 
 
    "clipping_epsilon_init_preference": 1e-5,
    "std_low_carbon_preference": 0.01, 

    "confirmation_bias": 1, 
    
    "init_carbon_price": 0, 
    "phi_lower": 0.02, 
    "BA_nodes": 11
    }
    tau_vals = [0.2,0.8]
    fileName = main(base_params=base_params, tau_vals = tau_vals)

    RUN_PLOT = 1

    if RUN_PLOT:
        timer_series_anti_heg_plot.main(fileName = fileName)
