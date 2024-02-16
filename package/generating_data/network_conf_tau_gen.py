# imports
import time
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object, generate_vals
from package.resources.run import emissions_parallel_run
from package.generating_data.twoD_param_sweep_gen import produce_param_list_stochastic_n_double, generate_vals_variable_parameters_and_norms


def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_2D.json",
        print_simu = 1,
        ) -> str: 

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
    property_min = var_params["property_min"]#0,
    property_max = var_params["property_max"]#1,
    property_reps = var_params["property_reps"]#10,
    property_varied_title = var_params["property_varied_title"]# #"A to Omega ratio"

    property_values_list = generate_vals(
        var_params
    )
    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    # load variable params
    f_variable_parameters = open(VARIABLE_PARAMS_LOAD)
    variable_parameters_dict = json.load(f_variable_parameters)
    f_variable_parameters.close()

    # AVERAGE OVER MULTIPLE RUNS
    variable_parameters_dict = generate_vals_variable_parameters_and_norms(
        variable_parameters_dict
    )

    root = "tax_sweep_networks"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    
    #Gen params lists
    #ENVIRONMENTAL IDENTITY
    params["alpha_change_state"] = "dynamic_identity_determined_weights"
    params["network_type"] = "SW"
    params_list_tax_SW_identity,__ = produce_param_list_stochastic_n_double(params, variable_parameters_dict)
    params["network_type"] = "SBM"
    params_list_tax_SBM_identity,__ = produce_param_list_stochastic_n_double(params, variable_parameters_dict)
    params["network_type"] = "BA"
    params_list_tax_BA_identity,__ = produce_param_list_stochastic_n_double(params, variable_parameters_dict)
    params_list_identity = params_list_tax_SW_identity + params_list_tax_SBM_identity + params_list_tax_BA_identity
    #SOCIAL
    params["alpha_change_state"] = "dynamic_socially_determined_weights"
    params["network_type"] = "SW"
    params_list_tax_SW_social,__ = produce_param_list_stochastic_n_double(params, variable_parameters_dict)
    params["network_type"] = "SBM"
    params_list_tax_SBM_social,__ = produce_param_list_stochastic_n_double(params, variable_parameters_dict)
    params["network_type"] = "BA"
    params_list_tax_BA_social,__ = produce_param_list_stochastic_n_double(params, variable_parameters_dict)
    params_list_social = params_list_tax_SW_social + params_list_tax_SBM_social + params_list_tax_BA_social
    
    params_list = params_list_identity + params_list_social
    print("Total runs: ",len(params_list))
    
    Data_serial = emissions_parallel_run(params_list)
    data_array = Data_serial.reshape(2,3,(variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"], params["seed_reps"]))
    #3 is for the networks, 2 is for the scenario

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    ##################################
    #save data

    createFolder(fileName)

    save_object(data_array, fileName + "/Data", "emissions_data_2_3")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_networks_conf_tau.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_2D_networks_conf_tau.json",
    )

