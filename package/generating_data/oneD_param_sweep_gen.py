"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_stock, multi_emissions_stock_ineq
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list_stochastic
from package.generating_data.twoD_param_sweep_gen import generate_vals_variable_parameters_and_norms

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_values_list 

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
        RUN_TYPE = 1
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

    root = "one_param_sweep_multi"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    params_list = produce_param_list_stochastic(params, property_values_list, property_varied)

    if RUN_TYPE == 1:
        emissions_stock_array, gini_list = multi_emissions_stock_ineq(params_list)
        emissions_array= emissions_stock_array.reshape(property_reps, params["seed_reps"])
        gini_array= gini_list.reshape(property_reps, params["seed_reps"])
    else:
        emissions_stock_array = multi_emissions_stock(params_list)
        emissions_array= emissions_stock_array.reshape(property_reps, params["seed_reps"])

    createFolder(fileName)

    save_object(emissions_array, fileName + "/Data", "emissions_array")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")
    if RUN_TYPE == 1:
        save_object(gini_array, fileName + "/Data", "gini_array")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_budget_ineq.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_budget_ineq.json",
        RUN_TYPE = 1
)

