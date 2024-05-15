
# imports
from ast import arg
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_stock
from package.resources.utility import produce_param_list_stochastic
from package.plotting_data import phi_effect_plot
from package.resources.utility import generate_vals
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
    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)


    root = "phi_vary"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    #DELTE
    #params["carbon_price_duration"] = 3
    
############################################################################################################################
    #SW
    params["network_type"]  = "SW"
    params["carbon_price_increased_lower"] = 0    
    params_list_no_tau_SW = produce_param_list_stochastic(params, property_values_list, property_varied)

    params["carbon_price_increased_lower"] = 0.1   
    params_list_low_tau_SW = produce_param_list_stochastic(params, property_values_list, property_varied)

    params["carbon_price_increased_lower"] = 1  
    params_list_high_tau_SW = produce_param_list_stochastic(params, property_values_list, property_varied)

    params_list_SW = params_list_no_tau_SW + params_list_low_tau_SW + params_list_high_tau_SW

##########################################################################################################
    #DELTE
    #params["carbon_price_duration"] = 3
    
    #SBM
    params["network_type"]  = "SBM"
    params["carbon_price_increased_lower"] = 0    
    params_list_no_tau_SBM = produce_param_list_stochastic(params, property_values_list, property_varied)

    params["carbon_price_increased_lower"] = 0.1  
    params_list_low_tau_SBM = produce_param_list_stochastic(params, property_values_list, property_varied)

    params["carbon_price_increased_lower"] = 1 
    params_list_high_tau_SBM = produce_param_list_stochastic(params, property_values_list, property_varied)

    params_list_SBM = params_list_no_tau_SBM + params_list_low_tau_SBM + params_list_high_tau_SBM

#########################################################################################################
    #DELTE
    #params["carbon_price_duration"] = 3
    
    #BA
    params["network_type"]  = "BA"
    params["carbon_price_increased_lower"] = 0
    params_list_no_tau_BA = produce_param_list_stochastic(params, property_values_list, property_varied)

    params["carbon_price_increased_lower"] = 0.1
    params_list_low_tau_BA = produce_param_list_stochastic(params, property_values_list, property_varied)

    params["carbon_price_increased_lower"] = 1
    params_list_high_tau_BA = produce_param_list_stochastic(params, property_values_list, property_varied)

    params_list_BA = params_list_no_tau_BA + params_list_low_tau_BA + params_list_high_tau_BA
##########################################################################################################

    #RUN THE STUFF
    params_list = params_list_SW + params_list_SBM + params_list_BA 
    print("TOTAL RUNS", len(params_list))

    emissions_stock_serial = multi_emissions_stock(params_list)
    emissions_array = emissions_stock_serial.reshape(3, 3, property_reps, params["seed_reps"])#2 is for BA and SBM,3 is for the 3 differents states
    
    emissions_array_SW = emissions_array[0]
    emissions_array_SBM = emissions_array[1]
    emissions_array_BA = emissions_array[2]

    #SAVE STUFF
    createFolder(fileName)
    
    save_object(emissions_array_BA , fileName + "/Data", "emissions_array_BA")
    save_object(emissions_array_SBM, fileName + "/Data", "emissions_array_SBM")
    save_object(emissions_array_SW, fileName + "/Data", "emissions_array_SW")
    save_object(params, fileName + "/Data", "base_params")
    save_object(params, fileName + "/Data", "base_params")
    save_object(params, fileName + "/Data", "base_params")

    print("RUNS DONE")

###############################################################################################################
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_networks_phi.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_networks_phi.json",
    )
    RUN_PLOT = 1

    if RUN_PLOT:
        phi_effect_plot.main(fileName = fileName_Figure_1)
