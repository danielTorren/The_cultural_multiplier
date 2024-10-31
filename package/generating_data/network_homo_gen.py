
# imports
import json
from package.resources.utility import createFolder,produce_name_datetime,save_object, produce_param_list_stochastic_multi_named
from package.resources.run import multi_emissions_stock
from package.resources.utility import generate_vals

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
        seed_labels = ["network_structure_seed", "shuffle_homophily_seed", "shuffle_coherance_seed"], 
        VARIABLE_RUNS = 1,
        FIXED_RUNS = 1
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

    root = "networks_homo_tau"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    

    homphily_states = [0, 0.8, 1]
    homophily_networks = ["SW", "SBM"]

    createFolder(fileName)

    save_object(homphily_states, fileName + "/Data", "homphily_states")
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")
    
    if VARIABLE_RUNS: 
    ##########################################################################################################
        #DO SBM AND SW
        params_list = []
        for i in homophily_networks:
            params["network_type"] = i
            for j in homphily_states:
                params["homophily_state"] = j
                param_list_specfic = produce_param_list_stochastic_multi_named(params, property_values_list, property_varied, seed_labels)
                params_list.extend(param_list_specfic)
    #########################################################################################################
        #BA
        params["network_type"] = "SF"
        #NO HEGEMONOY AND COMPLETE MIXING
        params["SF_green_or_brown_hegemony"] = 0    
        params["homophily_state"] = 0
        params_list_no_heg_SF = produce_param_list_stochastic_multi_named(params, property_values_list, property_varied, seed_labels)

        #Green HEGEMONOY AND homophily
        params["SF_green_or_brown_hegemony"] = 1   
        params["homophily_state"] = 1
        params_list_green_heg_SF = produce_param_list_stochastic_multi_named(params, property_values_list, property_varied, seed_labels)

        #Brown HEGEMONOY AND homophily
        params["SF_green_or_brown_hegemony"] = -1   
        params["homophily_state"] = 1
        params_list_brown_heg_SF = produce_param_list_stochastic_multi_named(params, property_values_list, property_varied, seed_labels)

        params_list_SF = params_list_no_heg_SF + params_list_green_heg_SF + params_list_brown_heg_SF

        params_list.extend(params_list_SF)
    ############################################################################################################################
        #RUN THE STUFF
    
        print("TOTAL RUNS", len(params_list))


        emissions_stock_serial = multi_emissions_stock(params_list)

        print("RUNS DONE")

        emissions_array = emissions_stock_serial.reshape(3, 3, property_reps, params["seed_reps"])#2 is for BA and SBM,3 is for the 3 differents states
        
        emissions_array_SW = emissions_array[0]
        emissions_array_SBM = emissions_array[1]
        emissions_array_SF = emissions_array[2]

        #SAVE STUFF
        save_object(emissions_array_SW, fileName + "/Data", "emissions_array_SW")
        save_object(emissions_array_SBM, fileName + "/Data", "emissions_array_SBM")
        save_object(emissions_array_SF , fileName + "/Data", "emissions_array_SF")
        
    if FIXED_RUNS:
        ########################################################################################################### 
        # ONLY NEED TO CALCULATE THE emissiosn for the property reps
        params["network_type"] = "SW"
        params["alpha_change_state"] = "fixed_preferences"
        fixed_params_list = produce_param_list_stochastic_multi_named(params, property_values_list, property_varied, seed_labels)

        print("TOTAL RUNS FIXED", len(fixed_params_list))


        fixed_emissions_stock_serial = multi_emissions_stock(fixed_params_list)
        fixed_emissions_array = fixed_emissions_stock_serial.reshape( len(property_values_list), params["seed_reps"])
        print("FIXED RUNS DONE")

        save_object(fixed_emissions_array, fileName + "/Data", "fixed_emissions_array")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_networks_homo_tax.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_networks_homo_tax.json",
        VARIABLE_RUNS = 1,
        FIXED_RUNS = 1
    )
