"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
from copy import deepcopy
import json
import numpy as np
from package.resources.utility import createFolder, produce_name_datetime, save_object
from package.resources.run import multi_burn_in_societies,multi_norm_emissions_load, multi_target_norm_emissions_load

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_values_list 

def produce_param_list_just_stochastic(params: dict) -> list[dict]:
    params_list = []
    for v in range(params["seed_reps"]):
        params["set_seed"] = int(v+1)
        params_list.append(
            params.copy()
        )  
    return params_list

def produce_param_list_emissions_target_just_stochastic(params,property_list, property, seed_list):
    params_list = []
    for i, val in enumerate(property_list):
        params[property] = val
        params["set_seed"] = seed_list[i]
        params_list.append(
                params.copy()
            )  
    return [params_list]#put it in brackets so its 2D like emissions target, params and stochastic case!

def produce_param_list_emissions_target_params_and_stochastic(params,property_list, property, target_list, target_property, seed_list):
    params_matrix = []

    for j, target in enumerate(target_list):
        params[target_property] = target
        params["set_seed"] = seed_list[j]
        params_row = []
        for i in property_list:
            params[property] = i
            params_row.append(
                    params.copy()
                )  
        params_matrix.append(params_row)
    return params_matrix

def calc_multiplier_matrix(vector_no_preference_change, matrix_preference_change_params):
    
    multiplier_matrix = 1 - matrix_preference_change_params/vector_no_preference_change
    return multiplier_matrix

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
        reduction_prop = 0.5,
        carbon_price_duration = 1000,
        tau_xtol = 1e-6
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

    root = "emission_target_sweep"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)


    ##################################
    #Gen burn in socieities for different seeds no carbon price, with preference change!!, Runs: seeds
    #OUTPUT: societies list array of length seed [socities_1, E_2,...,E_seed]
    #params["alpha_change"] = "static_preferences"
    params["carbon_price_duration"] =  0#no carbon price duration, only burn in
    params["carbon_price_increased"] = 0
    params["ratio_preference_or_consumption"] = 0#doesnt matter as its not used
    params["alpha_change"] = "dynamic_culturally_determined_weights"#burn in we do want preference change to stabalize system
    params_list_no_price_no_preference_change = produce_param_list_just_stochastic(params)
    seed_list = [x["set_seed"] for x in params_list_no_price_no_preference_change]
    
    #print("params_list_no_price_no_preference_change",params_list_no_price_no_preference_change)
    #print("seed_list",seed_list)
    
    societies_list = multi_burn_in_societies(params_list_no_price_no_preference_change)
    #print("Gen seeds no carbon price, no preference change, emissions_stock_seeds",emissions_stock_seeds)
    #print("societies_list",societies_list)
    stock_emissions_list = [x.total_carbon_emissions_stock for x in societies_list]
    print("stock_emissions_list shoudl be zero",stock_emissions_list)
    print("Societies burn in done")

    #quit()

    ##################################
    #Now run the burn in simulation with no carbon price for the carbon price time to establish the baseline, Runs: seeds*R
    #OUTPUT: TOTAL EMISSIONS for run for each seed, [E_1, E_2,...,E_seed]

    #take the models, make copies and the run them
    societies_model_no_price_no_preference_change_list = []
    for i, model_seed in enumerate(societies_list):
        model_seed.t = 0#reset time!
        model_seed.burn_in_duration = 0
        model_seed.carbon_price_duration = carbon_price_duration#set the carbon price duration, with no burn in period
        model_seed.carbon_price_increased = 0#need to run as if the carbon price was zero
        model_seed.switch_from_dynamic_to_static_preferences()#change from dynamic to static preferences, this is to avoid doing unnecessary calculations
        societies_model_no_price_no_preference_change_list.append(deepcopy(model_seed))

    norm_emissions_stock_seeds = multi_norm_emissions_load(societies_model_no_price_no_preference_change_list)
    print("emissions_stock_seeds after running for carbon tax time",norm_emissions_stock_seeds)
    print("Emissiosn targets calculated")
    #quit()
    ##################################
    #Calc the target emissions for each seed

    norm_emissions_target_seeds = norm_emissions_stock_seeds*reduction_prop

    ##################################
    #Gen seeds recursive carbon price, no preference change, Runs: seeds*R
    #OUTPUT: Carbon price for run for each seed, [tau_1, tau_2,...,tau_seed]

    #take the models, make copies and the run them
    societies_model_targect_no_preference_change_list = []
    for i, model_seed in enumerate(societies_list):
        model_seed.t = 0#reset time!
        model_seed.burn_in_duration = 0
        model_seed.carbon_price_duration = carbon_price_duration#set the carbon price duration, with no burn in period
        model_seed.norm_emissions_stock_target = norm_emissions_target_seeds[i]#set the emissions target reduction
        model_seed.switch_from_dynamic_to_static_preferences()#change from dynamic to static preferences, this is to avoid doing unnecessary calculations
        societies_model_targect_no_preference_change_list.append(deepcopy(model_seed))

    tau_seeds_no_preference_change_array = multi_target_norm_emissions_load([societies_model_targect_no_preference_change_list],tau_xtol)#put in brackets so its the same dimensions as later on!
    tau_seeds_no_preference_change = tau_seeds_no_preference_change_array[0]#This is cos we want to use the same functions for the latter 2d version so take the 0th element in the 2d list
    print("tau_seeds_no_preference_change",tau_seeds_no_preference_change)

    #quit()

    ##################################
    #Gen seeds recursive carbon price, preference change, with varying parameters, Runs: seeds*N*R
    #OUTPUT: Carbon price for run for each seed and parameters, [[tau_1_1, tau_2_1,...,tau_seed_1],..,[...,tau_seed_N]]
    
    #THERE HAS TO BE A SMARTER WAY TO DO THIS
    societies_model_targect_preference_change_list = []
    for i, model_seed in enumerate(societies_list):
        model_seed.t = 0#reset time!
        model_seed.burn_in_duration = 0
        model_seed.carbon_price_duration = carbon_price_duration
        model_seed.norm_emissions_stock_target = norm_emissions_target_seeds[i]
        #model_seed.alpha_change = "dynamic_culturally_determined_weights"#It should already be dynamic but just in case
        same_seed_list = []
        for j in property_values_list: 
            setattr(model_seed, property_varied, j)#BETTER THAN EVAL!
            same_seed_list.append(deepcopy(model_seed))
        societies_model_targect_preference_change_list.append(same_seed_list)

    #params["alpha_change"] = "dynamic_culturally_determined_weights"
    #params_list_emissions_target_preference_change = produce_param_list_emissions_target_params_and_stochastic(params,property_values_list, property_varied,emissions_target_seeds, "norm_emissions_stock_target", seed_list)
    #print("params_list_emissions_target_preference_change",params_list_emissions_target_preference_change)
    tau_seeds_preference_change = multi_target_norm_emissions_load(societies_model_targect_preference_change_list,tau_xtol)
    tau_seeds_preference_change_matrix_not_T = tau_seeds_preference_change.reshape(params["seed_reps"],property_reps)
    
    tau_seeds_preference_change_matrix = tau_seeds_preference_change_matrix_not_T.T  #take transpose so that the stuff seeds are back in the correct place!
    print("tau_seeds_preference_change_matrix",tau_seeds_preference_change_matrix)
    
    ##################################
    #Calc the multiplier for each seed and parameter: seed*N matrix

    multiplier_matrix = calc_multiplier_matrix(tau_seeds_no_preference_change, tau_seeds_preference_change_matrix)
    print("multiplier_matrix",multiplier_matrix)

    ##################################
    #save data

    createFolder(fileName)

    save_object(norm_emissions_stock_seeds, fileName + "/Data", "emissions_stock_seeds")
    save_object(tau_seeds_no_preference_change, fileName + "/Data", "tau_seeds_no_preference_change")
    save_object(tau_seeds_preference_change_matrix, fileName + "/Data", "tau_seeds_preference_change_matrix")
    save_object(multiplier_matrix, fileName + "/Data", "multiplier_matrix") 
    save_object(params, fileName + "/Data", "base_params")
    save_object(reduction_prop, fileName + "/Data", "reduction_prop")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_mu_target.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_mu_target.json",
        reduction_prop = 0.5,
        carbon_price_duration = 1000
)

