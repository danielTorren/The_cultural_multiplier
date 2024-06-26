# imports
import json
from package.resources.utility import createFolder,produce_name_datetime,save_object, generate_vals # produce_param_list_stochastic,
from package.resources.run import emissions_parallel_run_BLOCKS

def produce_param_list_stochastic(params: dict, property_list: list, property: str) -> list[dict]:
    params_list = []
    for i in property_list:
        params[property] = i
        for v in range(params["seed_reps"]):
            params["set_seed"] = int(v+1)
            params_list.append(
                params.copy()
            )  
    return params_list

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

    root = "SBM_BLOCK_sub_tau_vary"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    ##########################################
    # BOTH BLOCKS LOW SUB
    params["low_carbon_substitutability_upper"] = 1.1  
    params["low_carbon_substitutability_lower"] = 1.1  
    #NO "homophily_state"
    params["homophily_state"] = 0    
    params_list_low_sub_no_homo = produce_param_list_stochastic(params, property_values_list, property_varied)

    #High "homophily_state"
    params["homophily_state"] = 1  
    params_list_low_sub_high_homo = produce_param_list_stochastic(params, property_values_list, property_varied)

    params_list_low_sub = params_list_low_sub_no_homo + params_list_low_sub_high_homo

    ##########################################
    #BOTH BLOCKS HIGH SUB
    params["low_carbon_substitutability_upper"] = 8  
    params["low_carbon_substitutability_lower"] = 8  
    #NO "homophily_state"
    params["homophily_state"] = 0    
    params_list_high_sub_no_homo = produce_param_list_stochastic(params, property_values_list, property_varied)

    #High "homophily_state"
    params["homophily_state"] = 1  
    params_list_high_sub_high_homo= produce_param_list_stochastic(params, property_values_list, property_varied)

    params_list_high_sub = params_list_high_sub_no_homo + params_list_high_sub_high_homo

    ##########################################
    #one block high the other low
    params["low_carbon_substitutability_upper"] = 1.1
    params["low_carbon_substitutability_lower"] = 8  
    #NO "homophily_state"
    params["homophily_state"] = 0    
    params_list_mixed_sub_no_homo = produce_param_list_stochastic(params, property_values_list, property_varied)

    #High "homophily_state" with block 1 being the high one(THIS IS SO THAT THE GREEN HAS HIGH SUB)
    params["homophily_state"] = 1  
    params["low_carbon_substitutability_upper"] = 8 
    params["low_carbon_substitutability_lower"] = 1.1  
    params_list_green_high_sub_high_homo = produce_param_list_stochastic(params, property_values_list, property_varied)

    #High "homophily_state" with block 2 being the high one(THIS IS SO THAT THE BROWN HAS HIGH SUB)
    params["homophily_state"] = 1  
    params["low_carbon_substitutability_upper"] = 1.1 
    params["low_carbon_substitutability_lower"] = 8  
    params_list_brown_high_sub_high_homo = produce_param_list_stochastic(params, property_values_list, property_varied)

    params_list_mixed_sub = params_list_mixed_sub_no_homo + params_list_green_high_sub_high_homo + params_list_brown_high_sub_high_homo

    #####################################################################

    params_list = params_list_low_sub + params_list_high_sub + params_list_mixed_sub
    print("TOTAL RUNS: ", len(params_list))

    emissions_stock_serial, emissions_blocks_serial = emissions_parallel_run_BLOCKS(params_list)
    emissions_array = emissions_stock_serial.reshape(7, property_reps, params["seed_reps"])#7 for the 7 different scenarios
    emissions_array_blocks = emissions_blocks_serial.reshape(7, property_reps, params["seed_reps"], params["SBM_block_num"])#7 for the 7 different scenarios

    createFolder(fileName)
    
    save_object(emissions_array, fileName + "/Data", "emissions_array")
    save_object(emissions_array_blocks, fileName + "/Data", "emissions_array_blocks")
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_SBM_block_tau.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_SBM_block_tau.json",
)

