"""
Run simulation to compare the results for different cultural and preference change scenarios
"""

# imports
import json
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_flow_stock_run
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list_stochastic
#from package.generating_data.twoD_param_sweep_gen import generate_vals_variable_parameters_and_norms
from package.generating_data.oneD_param_sweep_gen import generate_vals
import numpy as np

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_consumption_ratio.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_consumption_ratio.json",
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

    root = "emisisons_flow_ratio_consumption"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    params_list = produce_param_list_stochastic(params, property_values_list, property_varied)

    time_array = np.arange(0,params["time_steps_max"] + params["compression_factor_state"],params["compression_factor_state"])

    #print("HEYEYU",params_list, len(params_list))
    __, emissions_flow_timeseries, __= multi_emissions_flow_stock_run(params_list)
    #print("emissions_flow_timeseries", emissions_flow_timeseries)
    emissions_flow_timeseries_array = emissions_flow_timeseries.reshape(property_reps, params["seed_reps"],len(time_array))

    createFolder(fileName)

    save_object(emissions_flow_timeseries_array, fileName + "/Data", "emissions_flow_timeseries_array")
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params, fileName + "/Data", "var_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")
    save_object(time_array, fileName + "/Data", "time_array")


    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_consumption_ratio.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_consumption_ratio.json",
)
