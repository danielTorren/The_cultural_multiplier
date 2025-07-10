import time
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object, generate_vals_2D, produce_param_list_stochastic_multi
from package.resources.run import emissions_parallel_run

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_networks_tau_ineq_alt.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_min_expenditure_share.json",
        print_simu = 1,
    ) -> str:

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    f_variable_parameters = open(VARIABLE_PARAMS_LOAD)
    variable_parameters_dict = json.load(f_variable_parameters)
    f_variable_parameters.close()

    root = "simple_h_min_expenditure"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    if print_simu:
        start_time = time.time()

    createFolder(fileName)
############################################################################################################
    # Core settings
    params["redistribution_state"] = 0
    params["expenditure_inequality_state"] = 0  # Still allow inequality to test interaction?
    params["alpha_change_state"] =  "fixed_preferences"
    networks_list = ["SW", "SBM", "SF"]
    params_list_ref = []

    for network in networks_list:
        params["network_type"] = network
        param_list_net_ref = produce_param_list_stochastic_multi(
            params,
            variable_parameters_dict["property_vals"],
            variable_parameters_dict["property_varied"]
        )
        params_list_ref.extend(param_list_net_ref)

    print("Total runs REF: ", len(params_list_ref))

    # Run the model
    Data_serial_ref = emissions_parallel_run(params_list_ref)

    data_array_ref = Data_serial_ref.reshape(len(networks_list), len(variable_parameters_dict["property_vals"]), params["seed_reps"])
    # Save results
    save_object(data_array_ref, fileName + "/Data", "emissions_data_min_expenditure_ref")

############################################################################################################
    # Core settings
    params["redistribution_state"] = 0
    params["expenditure_inequality_state"] = 0  # Still allow inequality to test interaction?
    params["alpha_change_state"] =  "dynamic_identity_determined_weights"
    networks_list = ["SW", "SBM", "SF"]
    params_list = []

    for network in networks_list:
        params["network_type"] = network
        param_list_net = produce_param_list_stochastic_multi(
            params,
            variable_parameters_dict["property_vals"],
            variable_parameters_dict["property_varied"]
        )
        params_list.extend(param_list_net)

    print("Total runs: ", len(params_list))

    # Run the model
    Data_serial = emissions_parallel_run(params_list)

    data_array = Data_serial.reshape(len(networks_list), len(variable_parameters_dict["property_vals"]), params["seed_reps"])
    # Save results
    save_object(data_array, fileName + "/Data", "emissions_data_min_expenditure")
    save_object(params, fileName + "/Data", "base_params")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")

    if print_simu:
        print("SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60))

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_simple_min_expen.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_simple_min_expen.json",
    )
