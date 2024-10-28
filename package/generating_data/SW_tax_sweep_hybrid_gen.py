# imports
import time
from copy import deepcopy
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object, produce_param_list_stochastic_multi
from package.resources.run import emissions_parallel_run
from package.resources.utility import generate_vals
from package.generating_data.network_multiplier_gen import main as calc_multiplier

def arrange_scenarios_tax(base_params_tax, carbon_tax_vals,scenarios):
    base_params_tax_copy = deepcopy(base_params_tax)
    params_list = []

    # 1. Run with fixed preferences, Emissions: [S_n]
    if "fixed_preferences" in scenarios:
        base_params_copy_1 = deepcopy(base_params_tax_copy)
        base_params_copy_1["alpha_change_state"] = "fixed_preferences"
        params_sub_list_1 = produce_param_list_stochastic_multi(base_params_copy_1, carbon_tax_vals,"carbon_price_increased")
        params_list.extend(params_sub_list_1)

    # 5. Run with social learning, Emissions: [S_n]
    if "dynamic_socially_determined_weights" in scenarios:
        base_params_copy_5 = deepcopy(base_params_tax_copy)
        base_params_copy_5["alpha_change_state"] = "dynamic_socially_determined_weights"
        params_sub_list_5 = produce_param_list_stochastic_multi(base_params_copy_5, carbon_tax_vals,"carbon_price_increased")
        params_list.extend(params_sub_list_5)

    # 6.  Run with cultural learning, Emissions: [S_n]
    if "dynamic_identity_determined_weights" in scenarios:
        base_params_copy_6 = deepcopy(base_params_tax_copy)
        base_params_copy_6["alpha_change_state"] = "dynamic_identity_determined_weights"
        params_sub_list_6 = produce_param_list_stochastic_multi(base_params_copy_6, carbon_tax_vals,"carbon_price_increased")
        params_list.extend(params_sub_list_6)
    
    if "hybrid_25" in scenarios:
        base_params_copy_7 = deepcopy(base_params_tax_copy)
        base_params_copy_7["alpha_change_state"] = "dynamic_hybrid_determined_weights"
        base_params_copy_7["M_identity"] = 2
        params_sub_list_7 = produce_param_list_stochastic_multi(base_params_copy_7, carbon_tax_vals,"carbon_price_increased")
        params_list.extend(params_sub_list_7)

    if "hybrid_50" in scenarios:
        base_params_copy_8 = deepcopy(base_params_tax_copy)
        base_params_copy_8["alpha_change_state"] = "dynamic_hybrid_determined_weights"
        base_params_copy_8["M_identity"] = 4
        params_sub_list_8 = produce_param_list_stochastic_multi(base_params_copy_8, carbon_tax_vals,"carbon_price_increased")
        params_list.extend(params_sub_list_8)

    if "hybrid_75" in scenarios:
        base_params_copy_9 = deepcopy(base_params_tax_copy)
        base_params_copy_9["alpha_change_state"] = "dynamic_hybrid_determined_weights"
        base_params_copy_9["M_identity"] = 6
        params_sub_list_9 = produce_param_list_stochastic_multi(base_params_copy_9, carbon_tax_vals,"carbon_price_increased")
        params_list.extend(params_sub_list_9)

    return params_list

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_tau_vary.json",
        print_simu = 1,
        scenarios = ["fixed_preferences", "dynamic_socially_determined_weights", "dynamic_identity_determined_weights" ],
        ) -> str: 

    scenario_reps = len(scenarios)

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
    property_reps = var_params["property_reps"]#10,
    property_varied_title = var_params["property_varied_title"]# #"A to Omega ratio"

    property_values_list = generate_vals(
        var_params
    )
    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)
    print("params", params)
    root = "SW_tax_sweep_hybrid"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    
    #Gen params lists
    params["network_type"] = "SW"
    params_list = arrange_scenarios_tax(params,property_values_list,scenarios)
    
    print("Total runs: ",len(params_list))
    
    Data_serial = emissions_parallel_run(params_list)
    data_array = Data_serial.reshape(scenario_reps , property_reps, params["seed_reps"] )

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    ##################################
    #save data

    createFolder(fileName)

    save_object(data_array, fileName + "/Data", "emissions_SW")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")
    save_object(scenarios, fileName + "/Data", "scenarios")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_SW_tax_sweep_hybrid.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_SW_tax_sweep_hybrid.json",
        scenarios = ["fixed_preferences","dynamic_socially_determined_weights", "dynamic_identity_determined_weights", "hybrid_25", "hybrid_50", "hybrid_75" ],
    )

