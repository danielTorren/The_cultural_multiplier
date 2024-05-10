# imports
import time
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object
from package.resources.run import emissions_parallel_run
from package.resources.utility import generate_vals
from package.generating_data.tax_sweep_BA_SBM_SW_gen import arrange_scenarios_tax

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_tau_vary.json",
        print_simu = 1,
        scenarios = ["fixed_preferences","uniform_network_weighting", "static_socially_determined_weights","static_culturally_determined_weights", "dynamic_socially_determined_weights", "dynamic_identity_determined_weights" ],
        ) -> str: 

    scenario_reps = len(scenarios)

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
    root = "asym_tax_sweep_networks"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    
    #Gen params lists
    params["network_type"] = "SW"
    params_list_tax_SW = arrange_scenarios_tax(params,property_values_list,scenarios)
    params["network_type"] = "SBM"
    params_list_tax_SBM = arrange_scenarios_tax(params,property_values_list,scenarios)
    params["network_type"] = "BA"
    params_list_tax_BA = arrange_scenarios_tax(params,property_values_list,scenarios)
    
    params_list = params_list_tax_SW + params_list_tax_SBM + params_list_tax_BA
    print("Total runs: ",len(params_list))
    
    Data_serial = emissions_parallel_run(params_list)
    data_array = Data_serial.reshape(3,scenario_reps , property_reps,params["seed_reps"])
    
    data_SW = data_array[0]
    data_SBM = data_array[1]
    data_BA = data_array[2]

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    ##################################
    #save data

    createFolder(fileName)

    save_object(data_SW, fileName + "/Data", "emissions_SW")
    save_object(data_SBM, fileName + "/Data", "emissions_SBM")
    save_object(data_BA, fileName + "/Data", "emissions_BA")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")
    save_object(scenarios, fileName + "/Data", "scenarios")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_networks_tau_vary_scenario_asym.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_networks_tau_vary_scenario.json",
        scenarios = ["fixed_preferences","uniform_network_weighting","static_socially_determined_weights","static_culturally_determined_weights","dynamic_socially_determined_weights", "dynamic_identity_determined_weights" ],
    )

