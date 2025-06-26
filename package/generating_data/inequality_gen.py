# imports
import time
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object, generate_vals_2D, produce_param_list_stochastic_n_double, produce_param_list_stochastic_multi
from package.resources.run import emissions_parallel_run_gini

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

    root = "network_ineq_tau"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    seeds_labels = ["preferences_seed", "network_structure_seed", "shuffle_homophily_seed", "shuffle_coherance_seed", "expenditure_seed"]
    #Gen params lists
    networks_list = ["SW","SBM", "SF"]

    createFolder(fileName)

    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    #NO REDISTRIBUTION ON ANY OF THE RUNS
    params["redistribution_state"] = 0
    params["expenditure_inequality_state"] = 0
    #RUN EQUALITY FOR COMPARISON
    params_list_ref = []
    for i in networks_list:
        params["network_type"] = i
        params_list_tax = produce_param_list_stochastic_multi(params, variable_parameters_dict["col"]["property_vals"], variable_parameters_dict["col"]["property_varied"])
        params_list_ref.extend(params_list_tax)

    print("Total runs REFERENCE: ",len(params_list_ref))

    Data_serial_ref, gini_serial_ref, poorest_spend_prop_serial_ref, richest_spend_prop_serial_ref = emissions_parallel_run_gini(params_list_ref)
    data_array_ref = Data_serial_ref.reshape(len(networks_list),variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    gini_array_ref =  gini_serial_ref.reshape(len(networks_list),variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    poorest_spend_prop_array_ref =  poorest_spend_prop_serial_ref.reshape(len(networks_list),variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    richest_spend_prop_array_ref =  richest_spend_prop_serial_ref.reshape(len(networks_list),variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    
    save_object(data_array_ref, fileName + "/Data", "emissions_data_networks_ref")
    save_object(gini_array_ref, fileName  + "/Data" , "gini_array_ref")
    save_object(poorest_spend_prop_array_ref, fileName  + "/Data" , "poorest_spend_prop_array_ref")
    save_object(richest_spend_prop_array_ref, fileName  + "/Data" , "richest_spend_prop_array_ref")

    
    print("DONE REFERENCE RUNS")


    #######################################################################################################################
    params["expenditure_inequality_state"] = 1
    params_list = []
    for i in networks_list:
        params["network_type"] = i
        params_list_tax = produce_param_list_stochastic_n_double(params, variable_parameters_dict, seeds_labels)
        params_list.extend(params_list_tax)

    print("Total runs: ",len(params_list))

    Data_serial, gini_serial, poorest_spend_prop_serial, richest_spend_prop_serial = emissions_parallel_run_gini(params_list)
    data_array = Data_serial.reshape(len(networks_list),variable_parameters_dict["row"]["property_reps"], variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    gini_array =  gini_serial.reshape(len(networks_list),variable_parameters_dict["row"]["property_reps"], variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    poorest_spend_prop_array =  poorest_spend_prop_serial.reshape(len(networks_list),variable_parameters_dict["row"]["property_reps"], variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    richest_spend_prop_array =  richest_spend_prop_serial.reshape(len(networks_list),variable_parameters_dict["row"]["property_reps"], variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    
    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    ##################################
    #save data
    save_object(data_array, fileName + "/Data", "emissions_data_networks")
    save_object(params, fileName + "/Data", "base_params")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    save_object(gini_array, fileName  + "/Data" , "gini_array")
    save_object(poorest_spend_prop_array, fileName  + "/Data" , "poorest_spend_prop_array")
    save_object(richest_spend_prop_array, fileName  + "/Data" , "richest_spend_prop_array")


    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    #WITH REDISTRIBUTION ON ANY OF THE RUNS
    params["redistribution_state"] = 1
    params["expenditure_inequality_state"] = 0

    #RUN EQUALITY FOR COMPARISON
    params_list_ref_with_re = []
    for i in networks_list:
        params["network_type"] = i
        params_list_tax_with_re = produce_param_list_stochastic_multi(params, variable_parameters_dict["col"]["property_vals"], variable_parameters_dict["col"]["property_varied"])
        params_list_ref_with_re.extend(params_list_tax_with_re)

    print("Total runs REFERENCE: ",len(params_list_ref_with_re))

    Data_serial_ref_with_re, gini_serial_ref_with_re, poorest_spend_prop_serial_ref_with_re, richest_spend_prop_serial_ref_with_re = emissions_parallel_run_gini(params_list_ref_with_re)
    data_array_ref_with_re = Data_serial_ref_with_re.reshape(len(networks_list),variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    gini_array_ref_with_re =  gini_serial_ref_with_re.reshape(len(networks_list),variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    poorest_spend_prop_array_ref_with_re =  poorest_spend_prop_serial_ref_with_re.reshape(len(networks_list),variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    richest_spend_prop_array_ref_with_re =  richest_spend_prop_serial_ref_with_re.reshape(len(networks_list),variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    
    save_object(data_array_ref_with_re, fileName + "/Data", "emissions_data_networks_ref_with_re")
    save_object(gini_array_ref_with_re, fileName  + "/Data" , "gini_array_ref_with_re")
    save_object(poorest_spend_prop_array_ref_with_re, fileName  + "/Data" , "poorest_spend_prop_array_re_with_ref")
    save_object(richest_spend_prop_array_ref_with_re, fileName  + "/Data" , "richest_spend_prop_array_ref_with_re")

    
    print("DONE REFERENCE RUNS WITH REDISTRIBUTION with redistribution")


    ######################################################################################################################
    params["expenditure_inequality_state"] = 1
    params_list_with_re = []
    for i in networks_list:
        params["network_type"] = i
        params_list_tax_with_re = produce_param_list_stochastic_n_double(params, variable_parameters_dict, seeds_labels)
        params_list_with_re.extend(params_list_tax_with_re)

    print("Total runs _with_re: ",len(params_list_with_re))

    Data_serial_with_re, gini_serial_with_re, poorest_spend_prop_serial_with_re, richest_spend_prop_serial_with_re = emissions_parallel_run_gini(params_list)
    data_array_with_re = Data_serial_with_re.reshape(len(networks_list),variable_parameters_dict["row"]["property_reps"], variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    gini_array_with_re =  gini_serial_with_re.reshape(len(networks_list),variable_parameters_dict["row"]["property_reps"], variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    poorest_spend_prop_array_with_re =  poorest_spend_prop_serial_with_re.reshape(len(networks_list),variable_parameters_dict["row"]["property_reps"], variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    richest_spend_prop_array_with_re =  richest_spend_prop_serial_with_re.reshape(len(networks_list),variable_parameters_dict["row"]["property_reps"], variable_parameters_dict["col"]["property_reps"], params["seed_reps"])
    
    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    ##################################
    #save data
    save_object(data_array_with_re, fileName + "/Data", "emissions_data_networks_with_re")
    save_object(params, fileName + "/Data", "base_params")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    save_object(gini_array_with_re, fileName  + "/Data" , "gini_array_with_re")
    save_object(poorest_spend_prop_array_with_re, fileName  + "/Data" , "poorest_spend_prop_array_with_re")
    save_object(richest_spend_prop_array_with_re, fileName  + "/Data" , "richest_spend_prop_array_with_re")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_networks_tau_ineq_alt.json",
        VARIABLE_PARAMS_LOAD = "package/constants/twoD_dict_networks_tau_ineq_alt_alt.json",
    )