
# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_stock,generate_data
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list_stochastic

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(np.log10(variable_parameters_dict["property_min"]),np.log10( variable_parameters_dict["property_max"]), variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_values_list 

def main(
        BASE_PARAMS_LOAD_BA = "package/constants/base_params.json",
        BASE_PARAMS_LOAD_SBM = "package/constants/base_params.json",
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

    f = open(BASE_PARAMS_LOAD_BA)
    params_BA = json.load(f)

    f = open(BASE_PARAMS_LOAD_SBM)
    params_SBM = json.load(f)

    root = "BA_SBM_tau_vary"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    
    createFolder(fileName)


#########################################################################################################
    #NO HEGEMONOY AND COMPLETE MIXING
    params_BA["BA_green_or_brown_hegemony"] = 0    
    params_BA["homophily_state"] = 0
    params_list_no_heg_BA = produce_param_list_stochastic(params_BA, property_values_list, property_varied)

    #Green HEGEMONOY AND homophily
    params_BA["BA_green_or_brown_hegemony"] = 1   
    params_BA["homophily_state"] = 1
    params_list_green_heg_BA = produce_param_list_stochastic(params_BA, property_values_list, property_varied)

    #Brown HEGEMONOY AND homophily
    params_BA["BA_green_or_brown_hegemony"] = -1   
    params_BA["homophily_state"] = 1
    params_list_brown_heg_BA = produce_param_list_stochastic(params_BA, property_values_list, property_varied)

    params_list_BA = params_list_no_heg_BA + params_list_green_heg_BA + params_list_brown_heg_BA
    print("TOTAL RUNS", len(params_list_BA))
    
    emissions_stock_serial_BA = multi_emissions_stock(params_list_BA)
    emissions_array_BA = emissions_stock_serial_BA.reshape(3, property_reps, params_BA["seed_reps"])#3 is for the 3 differents states

    save_object(emissions_array_BA, fileName + "/Data", "emissions_array_BA")
    save_object(params_BA, fileName + "/Data", "base_params_BA")
    print("BA DONE")
##########################################################################################################
    
    #NO "homophily_state"
    params_SBM["homophily_state"] = 0    
    params_list_no_tau_SBM = produce_param_list_stochastic(params_SBM, property_values_list, property_varied)

    #Low "homophily_state"
    params_SBM["homophily_state"] = 0.5   
    params_list_low_tau_SBM = produce_param_list_stochastic(params_SBM, property_values_list, property_varied)

    #High "homophily_state"
    params_SBM["homophily_state"] = 1  
    params_list_high_tau_SBM = produce_param_list_stochastic(params_SBM, property_values_list, property_varied)

    params_list_SBM = params_list_no_tau_SBM + params_list_low_tau_SBM + params_list_high_tau_SBM
    print("TOTAL RUNS", len(params_list_SBM))
    
    emissions_stock_serial_SBM = multi_emissions_stock(params_list_SBM)
    emissions_array_SBM = emissions_stock_serial_SBM.reshape(3, property_reps, params_SBM["seed_reps"])#3 is for the 3 differents states

    save_object(emissions_array_SBM, fileName + "/Data", "emissions_array_SBM")
    save_object(params_SBM, fileName + "/Data", "base_params_SBM")
    print("SBM DONE")
#####################################################################################################################
#CONSIDER REPLACING THIS WITH THE FUNCTIONS
#REFERENCE CASE NO SOCIAL EFFECTS - THESE SHOULD BE IDENTICAL BUT CHECK THIS!
#BA
    params_BA["BA_green_or_brown_hegemony"] = 0    
    params_BA["homophily_state"] = 0
    params_BA["alpha_change_state"] = "fixed_preferences"
    params_BA["phi"] = 0 #double sure!
    params_BA["seed_reps"] = 1
    params_list_no_heg_no_phi_BA = produce_param_list_stochastic(params_BA, property_values_list, property_varied)
    
    emissions_stock_BA = np.asarray(multi_emissions_stock(params_list_no_heg_no_phi_BA))
    save_object(emissions_stock_BA, fileName + "/Data", "emissions_array_BA_static")

#SBM
    params_SBM["homophily_state"] = 0
    params_SBM["alpha_change_state"] = "fixed_preferences"
    params_SBM["phi"] = 0 #double sure!
    params_SBM["seed_reps"] = 1
    params_list_no_heg_no_phi_SBM = produce_param_list_stochastic(params_SBM, property_values_list, property_varied)
    
    emissions_stock_SBM = np.asarray(multi_emissions_stock(params_list_no_heg_no_phi_SBM))
    save_object(emissions_stock_SBM, fileName + "/Data", "emissions_array_SBM_static")
    print("STATIC DONE")
########################################################################################################################
#REFERENCE CASE, helps with the base values
    params_SBM["carbon_price_increased"] = 1
    params_SBM["carbon_price_duration"] = 3
    reference_run = generate_data(params_SBM)  # run the simulation
    save_object(reference_run, fileName + "/Data", "reference_run")

###############################################################################################################
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD_BA = "package/constants/base_params_BA_tau.json",
        BASE_PARAMS_LOAD_SBM = "package/constants/base_params_SBM_tau.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_BA_SBM_tau.json",
)

