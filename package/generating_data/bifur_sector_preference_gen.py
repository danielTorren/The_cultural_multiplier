"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import preferences_parallel_run
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(np.log10(variable_parameters_dict["property_min"]),np.log10( variable_parameters_dict["property_max"]), variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_values_list 

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json"
         ) -> str: 

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    #print("params", var_params)

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
    property_reps = var_params["property_reps"]#10,

    property_values_list = generate_vals(
        var_params
    )
    #property_values_list = np.asarray([1.001,1.005,1.01,1.05, 1.1,1.5, 2, 2.5, 5,10, 15, 100])
    property_values_list = np.asarray([1.1e-1,1.1e0,1.2e0,1.5e0,1.1e1])
    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "bifur_one_param_sweep"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    
 

    # look at splitting of the last behaviour with preference dissonance
    params_list = produce_param_list(params, property_values_list, property_varied)
    data_array = preferences_parallel_run(params_list)

    createFolder(fileName)
    
    save_object(data_array, fileName + "/Data", "data_array")

    
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_bifur_sigma.json",#"package/constants/base_params_bifur_a.json",#"package/constants/base_params_bifur_seed.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_bifur_sigma.json"#"package/constants/oneD_dict_bifur_ratio_preference_or_consumption_state.json",#"package/constants/oneD_dict_bifur_a.json",#"package/constants/oneD_dict_bifur_seed.json"
)

