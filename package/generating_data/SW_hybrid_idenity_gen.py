# imports
import time
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object, generate_vals_2D, produce_param_list_stochastic_n_double
from package.resources.run import emissions_parallel_run

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_2D.json",
        print_simu = 1,
        ) -> str: 

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    # load variable params
    f_variable_parameters = open(VARIABLE_PARAMS_LOAD)
    variable_parameters_dict = json.load(f_variable_parameters)
    f_variable_parameters.close()

    # AVERAGE OVER MULTIPLE RUNS
    variable_parameters_dict = generate_vals_2D(variable_parameters_dict)

    root = "SW_M_M_identity"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    seeds_labels = ["preferences_seed", "network_structure_seed", "shuffle_homophily_seed", "shuffle_coherance_seed"]
    #Gen params lists

    params_list_tax = produce_param_list_stochastic_n_double(params, variable_parameters_dict, seeds_labels)

    print("Total runs: ",len(params_list_tax))

    Data_serial = emissions_parallel_run(params_list_tax)

    data_array = Data_serial.reshape(variable_parameters_dict["row"]["property_reps"], variable_parameters_dict["col"]["property_reps"], params["seed_reps"])

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    ##################################
    #save data

    createFolder(fileName)

    save_object(data_array, fileName + "/Data", "emissions_data")
    save_object(params, fileName + "/Data", "base_params")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_M_M_identity.json",
        VARIABLE_PARAMS_LOAD = "package/constants/twoD_dict_M_M_identity.json",
    )