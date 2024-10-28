# imports
import time
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object, generate_vals_2D
from package.resources.run import emissions_parallel_run
import numpy as np

def produce_param_list_stochastic_n_double_M(params_dict: dict, variable_parameters_dict: dict, seeds_labels) -> list[dict]:
    """
    Generate parameter list based on valid combinations of M and M_identity,
    ensuring M_identity <= M as specified by variable_parameters_dict.
    
    Parameters:
        params_dict (dict): Base parameters to modify.
        variable_parameters_dict (dict): Dictionary containing variable parameters, including valid M and M_identity combinations.
        seeds_labels (list): List of seed labels for stochasticity.
        
    Returns:
        list: List of dictionaries, each with a unique set of parameters for simulation.
    """
    params_list = []
    valid_combinations = variable_parameters_dict["M_M_identity_combinations"]

    for M, M_identity in valid_combinations:
        params_dict[variable_parameters_dict["col"]["property_varied"]] = M
        params_dict[variable_parameters_dict["row"]["property_varied"]] = M_identity
        
        for k in range(params_dict["seed_reps"]):
            for l, label in enumerate(seeds_labels):
                params_dict[label] = int(10 * l + k + 1)
            params_list.append(params_dict.copy())
    
    return params_list

def generate_M_M_identity_combinations(variable_parameters_dict):
    """
    Generate M and M_identity combinations based on the proportion rule,
    ensuring M_identity <= M by calculating M_identity as a proportion of M.
    
    Parameters:
        variable_parameters_dict (dict): Dictionary containing parameters to vary.
        
    Returns:
        list: List of tuples (M, M_identity) satisfying M_identity <= M.
    """
    # Generate M values
    M_values = np.round(np.linspace(
        variable_parameters_dict["col"]["property_min"], 
        variable_parameters_dict["col"]["property_max"], 
        variable_parameters_dict["col"]["property_reps"]
    ))

    # Generate M_identity proportions
    M_identity_values = np.round(np.linspace(
        variable_parameters_dict["row"]["property_min"], 
        variable_parameters_dict["row"]["property_max"], 
        variable_parameters_dict["row"]["property_reps"]
    ))

    # Generate combinations where M_identity is a proportion of M
    combinations = []
    for M in M_values:
        for M_identity in M_identity_values:
            if M_identity <= M:
                combinations.append((M, M_identity))
    return combinations

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_2D.json",
        print_simu = 1,
        ) -> str: 

    # Load base parameters
    with open(BASE_PARAMS_LOAD) as f:
        params = json.load(f)

    # Load variable parameters
    with open(VARIABLE_PARAMS_LOAD) as f_variable_parameters:
        variable_parameters_dict = json.load(f_variable_parameters)

    # Generate values for the 2D parameter space
    variable_parameters_dict = generate_vals_2D(variable_parameters_dict)

    # Generate M and M_identity combinations
    valid_combinations = generate_M_M_identity_combinations(variable_parameters_dict)
    variable_parameters_dict["M_M_identity_combinations"] = valid_combinations

    root = "SW_M_M_identity"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()

    seeds_labels = ["preferences_seed", "network_structure_seed", "shuffle_homophily_seed", "shuffle_coherance_seed"]

    # Generate parameter list with valid M and M_identity combinations
    params_list_tax = produce_param_list_stochastic_n_double_M(params, variable_parameters_dict, seeds_labels)

    print("Total runs: ", len(params_list_tax))

    Data_serial = emissions_parallel_run(params_list_tax)
    data_array = Data_serial.reshape(len(valid_combinations), params["seed_reps"] )
    
    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    # Save data
    createFolder(fileName)
    save_object(Data_serial , fileName + "/Data", "emissions_data")
    save_object(params, fileName + "/Data", "base_params")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_M_M_identity.json",
        VARIABLE_PARAMS_LOAD = "package/constants/twoD_dict_M_M_identity.json",
    )
