import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import interp1d
import json
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import multi_emissions_stock_flow_end
from package.generating_data.mu_sweep_carbon_price_gen import produce_param_list_stochastic
from package.generating_data.oneD_param_sweep_gen import generate_vals

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_comparison_runs.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_price.json",
        PHI_PARAMS_LOAD = "package/constants/oneD_dict_scenarios.json",
         ) -> str: 

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    #vary the tau value 
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

    f_phi = open(PHI_PARAMS_LOAD)
    phi_list = json.load(f_phi) 

    root = "deriving_multipliers"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    #social multiplier
    data_holder_social_multiplier = []
    params_list_social_multiplier= []
    params["alpha_change"] = "behavioural_independence"
    for i in phi_list:
        params["phi"] = i
        params_sub_list = produce_param_list_stochastic(params, property_values_list, property_varied)
        params_list_social_multiplier.extend(params_sub_list)#append to an emply list!

    #is_work = [(x["phi"],x["carbon_price_increased"]) for x in params_list_social_multiplier]
    #print("is_work", is_work)
    #quit()
    emissions_stock_social_multiplier, emissions_flow_social_multiplier = multi_emissions_stock_flow_end(params_list_social_multiplier)
    data_holder_stock_social_multiplier = emissions_stock_social_multiplier.reshape(len( phi_list ),property_reps, params["seed_reps"])
    data_holder_flow_social_multiplier = emissions_flow_social_multiplier.reshape(len( phi_list ),property_reps, params["seed_reps"])
    

    #print("data_holder_social_multiplier",data_holder_social_multiplier)

    #cultural_multiplier
    data_holder_cultural_multiplier = []
    params_list_cultural_multiplier= []
    params["alpha_change"] = "dynamic_culturally_determined_weights"
    for i in phi_list:
        params["phi"] = i
        params_sub_list = produce_param_list_stochastic(params, property_values_list, property_varied)
        params_list_cultural_multiplier.extend(params_sub_list)#append to an emply list!

    emissions_stock_cultural_multiplier, emissions_flow_cultural_multiplier = multi_emissions_stock_flow_end(params_list_cultural_multiplier)
    data_holder_stock_cultural_multiplier = emissions_stock_cultural_multiplier.reshape(len( phi_list ),property_reps, params["seed_reps"])
    data_holder_flow_cultural_multiplier = emissions_flow_cultural_multiplier.reshape(len( phi_list ),property_reps, params["seed_reps"])

    createFolder(fileName)

    save_object(data_holder_stock_social_multiplier, fileName + "/Data", "data_holder_stock_social_multiplier")
    save_object(data_holder_flow_social_multiplier, fileName + "/Data", "data_holder_flow_social_multiplier")
    save_object(data_holder_stock_cultural_multiplier, fileName + "/Data", "data_holder_stock_cultural_multiplier")
    save_object(data_holder_flow_cultural_multiplier, fileName + "/Data", "data_holder_flow_cultural_multiplier")
    save_object(params, fileName + "/Data", "base_params")
    save_object(phi_list, fileName + "/Data", "phi_list")
    save_object(var_params, fileName + "/Data", "var_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_deriving_multiplier.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_deriving_multiplier_price.json",
        PHI_PARAMS_LOAD = "package/constants/oneD_dict_deriving_multiplier_phi.json",
    )
