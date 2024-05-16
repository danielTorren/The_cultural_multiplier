
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
    #print("property_values_list", property_values_list)
    #quit()
    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)


    root = "phi_vary"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    carbon_price_list = [0, 0.1, 1]
    network_types = ["SW", "SBM", "BA"]
    params_list = []

    for network_type in network_types:
        params["network_type"] = network_type
        for carbon_price in carbon_price_list:
            params["carbon_price_increased_lower"] = carbon_price
            params_list += produce_param_list_stochastic(params, property_values_list, property_varied)

    print("TOTAL RUNS", len(params_list))
    emissions_stock_serial = multi_emissions_stock(params_list)
    emissions_array = emissions_stock_serial.reshape(len(network_types), len(carbon_price_list), property_reps, params["seed_reps"])#2 is for BA and SBM,3 is for the 3 differents states

    print("RUNS DONE")
########################################################################################################### 
    # I ONLY NEED TO CALCULATE THE emissiosn for the propoerty reps
    params["alpha_change_state"] = "fixed_preferences"
    params_list_fixed = []
    for carbon_price in carbon_price_list:
        params["carbon_price_increased_lower"] = carbon_price
        params_list_fixed += produce_param_list_only_stochastic(params)

    print("TOTAL RUNS FIXED", len(params_list_fixed))

    fixed_emissions_stock_serial = multi_emissions_stock(params_list_fixed)
    fixed_emissions_array = fixed_emissions_stock_serial.reshape( len(carbon_price_list), params["seed_reps"])
    print("FIXED RUNS DONE")
    ########################################################################################################### 
    #SAVE STUFF
    createFolder(fileName)
    
    save_object(emissions_array , fileName + "/Data", "emissions_array")
    save_object(fixed_emissions_array, fileName + "/Data", "fixed_emissions_array")
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")
    save_object(carbon_price_list,fileName + "/Data", "carbon_price_list" )
    #save_object(preferences_init_serial, fileName + "/Data", "preferences_init_serial")

    ###########################################################################################################


    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_networks_phi.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_networks_phi.json",
    )

