"""
Run simulation to compare the results for different cultural and preference change scenarios
"""

# imports
import json
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_stock
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list_stochastic
#from package.generating_data.twoD_param_sweep_gen import generate_vals_variable_parameters_and_norms
from package.generating_data.oneD_param_sweep_gen import generate_vals

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_comparison_runs.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_price.json",
        SCENARIOS_PARAMS_LOAD = "package/constants/oneD_dict_scenarios.json",
         ) -> str: 

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption",
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

    f_scenarios = open(SCENARIOS_PARAMS_LOAD)
    scenarios = json.load(f_scenarios) 

    root = "scenario_comparison"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    # Attitude BASED
    data_holder = []
    params_list= []
    params["ratio_preference_or_consumption"] = 1.0
    for i in scenarios:
        params["alpha_change"] = i
        params_sub_list = produce_param_list_stochastic(params, property_values_list, property_varied)
        params_list.extend(params_sub_list)#append to an emply list!

    #print("HEYEYU",params_list)
    emissions_stock_array, init_emissions_stock_array = multi_emissions_stock(params_list)
    data_holder = emissions_stock_array.reshape(len(scenarios),property_reps, params["seed_reps"])
    init_data_holder = init_emissions_stock_array.reshape(len(scenarios),property_reps, params["seed_reps"])

    # CONSUMPTION BASED
    data_holder_consumption_based = []
    params_list_consumption_based = []
    params["ratio_preference_or_consumption"] = 0.0
    for i in scenarios:
        params["alpha_change"] = i
        params_sub_list = produce_param_list_stochastic(params, property_values_list, property_varied)
        params_list_consumption_based.extend(params_sub_list)
    #print("adfsdf", params_list_consumption_based)
    emissions_stock_array_consumption_based, init_emissions_stock_array_consumption_based= multi_emissions_stock(params_list_consumption_based)
    data_holder_consumption_based = emissions_stock_array_consumption_based.reshape(len(scenarios),property_reps, params["seed_reps"])
    init_data_holder_consumption_based = init_emissions_stock_array_consumption_based.reshape(len(scenarios),property_reps, params["seed_reps"])

    createFolder(fileName)

    save_object(data_holder, fileName + "/Data", "data_holder")
    save_object(data_holder_consumption_based, fileName + "/Data", "data_holder_consumption_based")
    save_object(init_data_holder, fileName + "/Data", "init_data_holder")
    save_object(init_data_holder_consumption_based, fileName + "/Data", "init_data_holder_consumption_based")
    save_object(params, fileName + "/Data", "base_params")
    save_object(scenarios, fileName + "/Data", "scenarios")
    save_object(var_params, fileName + "/Data", "var_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")


    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_comparison_runs.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_price.json",
        SCENARIOS_PARAMS_LOAD = "package/constants/oneD_dict_scenarios.json",
)
