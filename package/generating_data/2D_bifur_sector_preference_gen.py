"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import preferences_parallel_run,preferences_consumption_parallel_run
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(np.log10(variable_parameters_dict["property_min"]),np.log10( variable_parameters_dict["property_max"]), variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_values_list 

def produce_param_list_2D_single_seed(params, property_values_list_1,property_values_list_2, property_varied_1, property_varied_2):

    params_list = []
    for i in property_values_list_1:
        params[property_varied_1] = i
        for j in property_values_list_2:
            params[property_varied_2] = j
            params_list.append(
                params.copy()
            )  # have to make a copy so that it actually appends a new dict and not just the location of the params dict
    return params_list

def produce_param_list_2D_stochastic_seed(params, property_values_list_1,property_values_list_2, property_varied_1, property_varied_2):

    params_list = []
    for i in property_values_list_1:
        params[property_varied_1] = i
        for j in property_values_list_2:
            params[property_varied_2] = j
            for j in range(params["seed_reps"]):
                params["set_seed"] = int(j+1)
                params_list.append(
                    params.copy()
                )  
    return params_list

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD_1 = "package/constants/variable_parameters_dict_SA.json",
        VARIABLE_PARAMS_LOAD_2 = "package/constants/variable_parameters_dict_SA.json",
        RUN_TYPE = 1
        ) -> str: 

    f_var_1 = open(VARIABLE_PARAMS_LOAD_1)
    var_params_1 = json.load(f_var_1) 

    f_var_2 = open(VARIABLE_PARAMS_LOAD_2)
    var_params_2 = json.load(f_var_2) 

    #print("params", var_params)

    property_varied_1 = var_params_1["property_varied"]#"ratio_preference_or_consumption_state",
    property_reps_1 = var_params_1["property_reps"]#10,

    property_values_list_1 = generate_vals(
        var_params_1
    )
    property_varied_2 = var_params_2["property_varied"]#"ratio_preference_or_consumption_state",
    property_reps_2 = var_params_2["property_reps"]#10,
    property_values_list_2 = generate_vals(
        var_params_2
    )

    print("property_values_list_1 ",property_values_list_1 )
    print("property_values_list_2 ",property_values_list_2)

    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "2D_bifur_param_sweep"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    

    # look at splitting of the last behaviour with preference dissonance
    
    if RUN_TYPE == 1:
        params_list = produce_param_list_2D_single_seed(params, property_values_list_1,property_values_list_2, property_varied_1, property_varied_2)
        print("Total Runs", len(params_list))
        data_array_serial = preferences_parallel_run(params_list)#shape is nxM, (property_reps_2*property_reps_1)*M
        data_array = data_array_serial.reshape(property_reps_1,property_reps_2, params["N"],params["M"])
    elif RUN_TYPE == 2:
        params_list = produce_param_list_2D_stochastic_seed(params, property_values_list_1,property_values_list_2, property_varied_1, property_varied_2)
        data_array_serial = preferences_parallel_run(params_list)#shape is nxM, (property_reps_2*property_reps_1*seed_reps)*M
        data_array = data_array_serial.reshape(property_reps_1,property_reps_2, params["seed_reps"], params["N"],params["M"])
    elif RUN_TYPE == 3:
        params_list = produce_param_list_2D_single_seed(params, property_values_list_1,property_values_list_2, property_varied_1, property_varied_2)
        print("Total Runs", len(params_list))
        data_array_serial, data_consumption_H_serial, data_consumption_L_serial, = preferences_consumption_parallel_run(params_list)#shape is nxM, (property_reps_2*property_reps_1)*M
        data_array = data_array_serial.reshape(property_reps_1,property_reps_2, params["N"],params["M"])
        data_array_H = data_consumption_H_serial.reshape(property_reps_1,property_reps_2, params["N"],params["M"])
        data_array_L = data_consumption_L_serial.reshape(property_reps_1,property_reps_2, params["N"],params["M"])

    
    createFolder(fileName)

    if RUN_TYPE == 3:
        save_object(data_array_H, fileName + "/Data", "data_array_H")
        save_object(data_array_L, fileName + "/Data", "data_array_L")

    save_object(data_array, fileName + "/Data", "data_array")
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params_1,fileName + "/Data" , "var_params_1")
    save_object(property_values_list_1,fileName + "/Data", "property_values_list_1")
    save_object(var_params_2,fileName + "/Data" , "var_params_2")
    save_object(property_values_list_2,fileName + "/Data", "property_values_list_2")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_bifur_sigma_ratio.json",#"package/constants/base_params_bifur_ratio_preference_or_consumption_state.json",#"package/constants/base_params_bifur_ratio_preference_or_consumption_state_and_a.json",#"package/constants/base_params_bifur_a.json",#"package/constants/base_params_bifur_seed.json",
        VARIABLE_PARAMS_LOAD_1 = "package/constants/oneD_dict_bifur_sigma.json",#"package/constants/oneD_dict_bifur_a.json",#"package/constants/oneD_dict_bifur_a.json",#"package/constants/oneD_dict_bifur_seed.json"
        VARIABLE_PARAMS_LOAD_2 = "package/constants/oneD_dict_bifur_ratio_preference_or_consumption_state.json",#"package/constants/oneD_dict_bifur_a.json",#"package/constants/oneD_dict_bifur_seed.json",
        RUN_TYPE = 3,
)

