
# imports
from ast import arg
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_stock
from package.resources.utility import produce_param_list_stochastic, produce_param_list_only_stochastic, generate_vals
##################################################################################################
#REVERSE Engineer the carbon price based on the final emissions

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
         ) -> str: 

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
    property_reps = var_params["property_reps"]#10,

    property_values_list = generate_vals(
        var_params
    )

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "fixed_preferences_tau_vary"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    nu_list = [1.01, 2, 5, 10, 20]
    params_list = []

    for nu in nu_list:
        params["sector_substitutability"] = nu
        params_list += produce_param_list_stochastic(params, property_values_list, property_varied)

    print("TOTAL RUNS", len(params_list))
    emissions_stock_serial = multi_emissions_stock(params_list)
    #print("emissions_stock_serial", emissions_stock_serial)
    
    emissions_array = emissions_stock_serial.reshape( len(nu_list), property_reps, params["seed_reps"])#2 is for BA and SBM,3 is for the 3 differents states

    
    print("RUNS DONE")

    ################################################################################# 
    #SAVE STUFF
    createFolder(fileName)
    
    save_object(emissions_array , fileName + "/Data", "emissions_array")
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")
    save_object(nu_list,fileName + "/Data", "nu_list" )

    ###########################################################################################################


    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_fixed_preferences_tau.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_fixed_preferences_tau.json",
    )

