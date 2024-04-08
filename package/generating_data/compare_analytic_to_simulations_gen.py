# imports
import time
from copy import deepcopy
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object
from package.resources.run import emissions_parallel_run, emissions_parallel_run_sectors
from copy import deepcopy
from package.resources.utility import generate_vals

def produce_param_list_scenarios_tax(params: dict, property_list: list, property: str) -> list[dict]:
    params_list = []
    for i in property_list:
        params[property] = i
        for v in range(params["seed_reps"]):
            params["set_seed"] = int(v+1)
            params_list.append(params.copy())  
    return params_list

def arrange_scenarios_tax(base_params_tax, carbon_tax_vals):
    
    """
    There are 2 scenarios here, 1.two sectos with where i have an assymetric carbon price that changes sector 1 2. Is where i have only 1 sector, but same budget
    """
    
    
    base_params_tax_copy = deepcopy(base_params_tax)
    params_list = []

    base_params_copy_1 = deepcopy(base_params_tax_copy)
    base_params_copy_1["M"] = 2
    params_sub_list_1 = produce_param_list_scenarios_tax(base_params_copy_1, carbon_tax_vals,"carbon_price_increased_lower")
    params_list.extend(params_sub_list_1)


    base_params_copy_2 = deepcopy(base_params_tax_copy)
    base_params_copy_1["M"] = 1
    params_sub_list_2 = produce_param_list_scenarios_tax(base_params_copy_2, carbon_tax_vals,"carbon_price_increased_lower")
    params_list.extend(params_sub_list_2)

    return params_list

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_tau_vary.json",
        print_simu = 1
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
    print("params", params)
    root = "compare_analytic_emissions"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()

    ############################################
    #NEED TO MAKE SURE THE CARBON PRICE IS ACTUALLY ASYMETRIC
    #case 1

    params["M"] = 1
    params["heterogenous_carbon_price_state"] = 0
    
    #Gen params lists
    params["network_type"] = "SW"
    params_list_tax_SW_1 = produce_param_list_scenarios_tax(params, property_values_list,"carbon_price_increased_lower")
    params["network_type"] = "SBM"
    params_list_tax_SBM_1 = produce_param_list_scenarios_tax(params, property_values_list,"carbon_price_increased_lower")
    params["network_type"] = "BA"
    params_list_tax_BA_1 = produce_param_list_scenarios_tax(params, property_values_list,"carbon_price_increased_lower")
    
    params_list_1 = params_list_tax_SW_1 + params_list_tax_SBM_1 + params_list_tax_BA_1

    ############################################
    #case 2

    params["M"] = 2
    params["carbon_price_increased_upper"] = 0
    params["heterogenous_carbon_price_state"] = 1

    #Gen params lists
    params["network_type"] = "SW"
    params_list_tax_SW_2 = produce_param_list_scenarios_tax(params, property_values_list,"carbon_price_increased_lower")
    params["network_type"] = "SBM"
    params_list_tax_SBM_2 = produce_param_list_scenarios_tax(params, property_values_list,"carbon_price_increased_lower")
    params["network_type"] = "BA"
    params_list_tax_BA_2 = produce_param_list_scenarios_tax(params, property_values_list,"carbon_price_increased_lower")
    
    params_list_2 = params_list_tax_SW_2 + params_list_tax_SBM_2 + params_list_tax_BA_2


    print("Total runs: ",len(params_list_1) + len(params_list_2))
    
    Data_serial_1, Data_serial_sectors_1  = emissions_parallel_run_sectors(params_list_1)
    data_array_1 = Data_serial_1.reshape(3, property_reps, params["seed_reps"])
    data_sectors_1 = Data_serial_sectors_1.reshape(3, property_reps, params["seed_reps"], 1)

    Data_serial_2, Data_serial_sectors_2  = emissions_parallel_run_sectors(params_list_2)
    data_array_2 = Data_serial_2.reshape(3, property_reps,params["seed_reps"])
    data_sectors_2 = Data_serial_sectors_2.reshape(3, property_reps, params["seed_reps"], 2)

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    ##################################
    #save data

    createFolder(fileName)

    save_object(data_array_1, fileName + "/Data", "data_array_1")
    save_object(data_array_2, fileName + "/Data", "data_array_2")
    save_object(data_sectors_1, fileName + "/Data", "data_sectors_1")
    save_object(data_sectors_2, fileName + "/Data", "data_sectors_2")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_networks_compare_analytic.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_asym_networks_compare_analytic.json"
    )

