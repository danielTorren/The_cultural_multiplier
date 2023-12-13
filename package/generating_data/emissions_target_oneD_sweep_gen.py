"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
import time
from copy import deepcopy
import json
import numpy as np
from package.resources.utility import createFolder, produce_name_datetime, save_object
from package.resources.run import multi_burn_in_societies,multi_emissions_load, multi_target_emissions_load

def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(np.log10(variable_parameters_dict["property_min"]),np.log10( variable_parameters_dict["property_max"]), variable_parameters_dict["property_reps"])
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
        print_simu = 1,
        static_weighting_run = 0
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

    root = "emission_target_sweep"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    
    ##################################
    #Gen burn in socieities for different seeds no carbon price, with preference change!!, Runs: seeds
    #OUTPUT: societies list array of length seed [socities_1, E_2,...,E_seed]
    #params["alpha_change_state"] = "fixed_preferences"
    params["carbon_price_duration"] =  0#no carbon price duration, only burn in
    params["carbon_price_increased"] = 0
    params["ratio_preference_or_consumption_state"] = 0 #WE ASSUME Consumption BASE LEARNING
    params["alpha_change_state"] = "dynamic_culturally_determined_weights"#burn in we do want preference change to stabalize system
    params_list_no_price_no_preference_change = produce_param_list_just_stochastic(params)
    #seed_list = [x["set_seed"] for x in params_list_no_price_no_preference_change]
    
    societies_list = multi_burn_in_societies(params_list_no_price_no_preference_change)
    #print("Gen seeds no carbon price, no preference change, emissions_stock_seeds",emissions_stock_seeds)
    #print("societies_list",societies_list)
    stock_emissions_list = [x.total_carbon_emissions_stock for x in societies_list]
    print("stock_emissions_list shoudl be zero",stock_emissions_list)
    print("Societies burn in done")

    #quit()


    """
    ##################################
    #Now run the burn in simulation with no carbon price for the carbon price time to establish the baseline, Runs: seeds*R
    #OUTPUT: TOTAL EMISSIONS for run for each seed, [E_1, E_2,...,E_seed]

    #take the models, make copies and the run them
    societies_model_no_price_no_preference_change_list = []
    for i, model_seed in enumerate(societies_list):
        model_copy = deepcopy(model_seed)
        model_copy.t = 0#reset time!
        model_copy.burn_in_duration = 0
        model_copy.carbon_price_duration = carbon_price_duration#set the carbon price duration, with no burn in period
        model_copy.carbon_price_increased = 0#need to run as if the carbon price was zero
        model_copy.switch_from_dynamic_to_fixed_preferences()#change from dynamic to static preferences, this is to avoid doing unnecessary calculations
        societies_model_no_price_no_preference_change_list.append(model_copy)

    emissions_stock_seeds = multi_emissions_load(societies_model_no_price_no_preference_change_list)
    print("emissions_stock_seeds after running for carbon tax time",emissions_stock_seeds)
    #print("Emissiosn targets calculated")
    """

    #########################################
    #Now run the burn in simulation with no carbon price for the carbon price time to establish the baseline, Runs: seeds*R
    #mu = 0, so preference change but no tax, CONSUMPTION BASED LEARNING
    #OUTPUT: TOTAL EMISSIONS for run for each seed, [E_1, E_2,...,E_seed]

    #take the models, make copies and the run them
    societies_model_no_price_preference_change_list = []
    for i, model_seed in enumerate(societies_list):
        model_copy = deepcopy(model_seed)
        #model_copy.t = 0#reset time!# not sure these are necessary
        #model_copy.burn_in_duration = 0
        model_copy.carbon_price_duration = carbon_price_duration#set the carbon price duration, with no burn in period
        model_copy.carbon_price_increased = 0#need to run as if the carbon price was zero
        #model_copy.switch_from_dynamic_to_fixed_preferences()#change from dynamic to static preferences, this is to avoid doing unnecessary calculations
        societies_model_no_price_preference_change_list.append(model_copy)

    emissions_stock_seeds = multi_emissions_load(societies_model_no_price_preference_change_list)
    emissions_target_seeds = emissions_stock_seeds*reduction_prop
    
    #Below also assume that the inital ratio between low and high carbon goods is 1, and both are 1?
    price_ratio = 1
    tau_guess = (1/reduction_prop) - price_ratio#ITS PROPORTIONAL TO THE SIZE OF THE REDUCTION IN EMISSIONS AND THE PRICE ratio of high and low carbon goods()
    
    print("emissions_stock_seeds",emissions_stock_seeds)
    
    ##################################
    #Gen seeds recursive carbon price, no preference change, Runs: seeds*R
    #OUTPUT: Carbon price for run for each seed, [tau_1, tau_2,...,tau_seed]
    #take the models, make copies and the run them
    societies_model_targect_no_preference_change_list = []
    for i, model_seed in enumerate(societies_list):
        model_copy = deepcopy(model_seed)
        #model_copy.t = 0#reset time!
        #model_copy.burn_in_duration = 0
        model_copy.carbon_price_duration = carbon_price_duration#set the carbon price duration, with no burn in period
        model_copy.emissions_stock_target = emissions_target_seeds[i]#set the emissions target reduction
        model_copy.switch_from_dynamic_to_fixed_preferences()#change from dynamic to static preferences, this is to avoid doing unnecessary calculations
        societies_model_targect_no_preference_change_list.append(model_copy)
    
    tau_seeds_no_preference_change = multi_target_emissions_load(societies_model_targect_no_preference_change_list,tau_guess)#put in brackets so its the same dimensions as later on!
    #tau_seeds_no_preference_change = tau_seeds_no_preference_change_array#This is cos we want to use the same functions for the latter 2d version so take the 0th element in the 2d list
    print("tau_seeds_no_preference_change",tau_seeds_no_preference_change)

    ##################################
    #Gen seeds recursive carbon price, ATTITUDE based preference  , Runs: seeds*R
    #OUTPUT: Carbon price for run for each seed, [tau_1, tau_2,...,tau_seed]

    #take the models, make copies and the run them
    societies_model_targect_attiude_preference_change_list = []
    for i, model_seed in enumerate(societies_list):
        model_copy = deepcopy(model_seed)
        model_copy.ratio_preference_or_consumption_state = 1.0
        #model_copy.t = 0#reset time!
        #model_copy.burn_in_duration = 0
        model_copy.carbon_price_duration = carbon_price_duration#set the carbon price duration, with no burn in period
        model_copy.emissions_stock_target = emissions_target_seeds[i]#set the emissions target reduction
        #model_copy.switch_from_dynamic_to_fixed_preferences()#change from dynamic to static preferences, this is to avoid doing unnecessary calculations
        societies_model_targect_attiude_preference_change_list.append(model_copy)
    
    tau_seeds_attitude_preference_change = multi_target_emissions_load(societies_model_targect_attiude_preference_change_list,tau_guess)#put in brackets so its the same dimensions as later on!
    #tau_seeds_no_preference_change = tau_seeds_no_preference_change_array#This is cos we want to use the same functions for the latter 2d version so take the 0th element in the 2d list
    print("tau_seeds_attitude_preference_change",tau_seeds_attitude_preference_change)

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    #Gen seeds recursive carbon price, CONSUMPTION BASED preference change, with varying parameters, Runs: seeds*N*R
    #OUTPUT: Carbon price for run for each seed and parameters, [[tau_1_1, tau_2_1,...,tau_seed_1],..,[...,tau_seed_N]]

    societies_model_targect_preference_change_list = []
    for i, model_seed in enumerate(societies_list):#loop through different seeds
        model_copy = deepcopy(model_seed)
        #model_copy.t = 0#reset time!
        #model_copy.burn_in_duration = 0
        model_copy.carbon_price_duration = carbon_price_duration
        model_copy.emissions_stock_target = emissions_target_seeds[i]
        #model_seed.alpha_change_state = "dynamic_culturally_determined_weights"#It should already be dynamic but just in case
        for j in property_values_list: #loop trough the different param values
            model_copy_param = deepcopy(model_copy)
            setattr(model_copy_param, property_varied, j)#BETTER THAN EVAL!
            societies_model_targect_preference_change_list.append(model_copy_param)

    #params["alpha_change_state"] = "dynamic_culturally_determined_weights"
    #params_list_emissions_target_preference_change = produce_param_list_emissions_target_params_and_stochastic(params,property_values_list, property_varied,emissions_target_seeds, "emissions_stock_target", seed_list)
    #print("params_list_emissions_target_preference_change",params_list_emissions_target_preference_change)
    
    tau_seeds_preference_change = multi_target_emissions_load(societies_model_targect_preference_change_list,tau_guess)
    
    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    
    tau_seeds_preference_change_matrix_not_T = tau_seeds_preference_change.reshape(params["seed_reps"],property_reps)
    
    tau_seeds_preference_change_matrix = tau_seeds_preference_change_matrix_not_T.T  #take transpose so that the stuff seeds are back in the correct place!
    print("tau_seeds_preference_change_matrix",tau_seeds_preference_change_matrix)
    
    ##################################
    #Calc the multiplier for each seed and parameter: seed*N matrix

    multiplier_matrix_attitude_preference_change = calc_multiplier_matrix(tau_seeds_attitude_preference_change, tau_seeds_preference_change_matrix)
    multiplier_matrix_no_preference_change = calc_multiplier_matrix(tau_seeds_no_preference_change, tau_seeds_preference_change_matrix)
    print("multiplier_matrix_attitude_preference_change",multiplier_matrix_attitude_preference_change)
    print("multiplier_matrix_no_preference_change",multiplier_matrix_no_preference_change)

    ##################################
    #save data

    createFolder(fileName)

    save_object(emissions_stock_seeds, fileName + "/Data", "emissions_stock_seeds")
    save_object(tau_seeds_no_preference_change, fileName + "/Data", "tau_seeds_no_preference_change")
    save_object(tau_seeds_preference_change_matrix, fileName + "/Data", "tau_seeds_preference_change_matrix")
    save_object(multiplier_matrix_attitude_preference_change, fileName + "/Data", "multiplier_matrix_attitude_preference_change") 
    save_object(multiplier_matrix_no_preference_change, fileName + "/Data", "multiplier_matrix_no_preference_change") 
    save_object(params, fileName + "/Data", "base_params")
    save_object(reduction_prop, fileName + "/Data", "reduction_prop")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")

    if static_weighting_run:
        ########################################################################################################################################
        #Gen seeds recursive carbon price, CONSUMPTION BASED preference change, but STATIC WEIGHTING, with varying parameters, Runs: seeds*N*R
        #OUTPUT: Carbon price for run for each seed and parameters, [[tau_1_1, tau_2_1,...,tau_seed_1],..,[...,tau_seed_N]]

        societies_model_targect_static_weighting_preference_change_list = []
        for i, model_seed in enumerate(societies_list):#loop through different seeds
            model_copy = deepcopy(model_seed)
            #model_copy.t = 0#reset time!
            #model_copy.burn_in_duration = 0
            model_copy.carbon_price_duration = carbon_price_duration
            model_copy.emissions_stock_target = emissions_target_seeds[i]
            model_copy.alpha_change_state = "static_culturally_determined_weights"
            #model_seed.alpha_change_state = "dynamic_culturally_determined_weights"#It should already be dynamic but just in case
            for j in property_values_list: #loop trough the different param values
                model_copy_param = deepcopy(model_copy)
                setattr(model_copy_param, property_varied, j)#BETTER THAN EVAL!
                societies_model_targect_static_weighting_preference_change_list.append(model_copy_param)

        #params["alpha_change_state"] = "dynamic_culturally_determined_weights"
        #params_list_emissions_target_preference_change = produce_param_list_emissions_target_params_and_stochastic(params,property_values_list, property_varied,emissions_target_seeds, "emissions_stock_target", seed_list)
        #print("params_list_emissions_target_preference_change",params_list_emissions_target_preference_change)
        
        tau_seeds_static_weighting_preference_change = multi_target_emissions_load(societies_model_targect_static_weighting_preference_change_list,tau_guess)
        tau_seeds_static_weighting_preference_change_matrix_not_T = tau_seeds_static_weighting_preference_change.reshape(params["seed_reps"],property_reps)
    
        tau_seeds_static_weighting_preference_change_matrix = tau_seeds_static_weighting_preference_change_matrix_not_T.T  #take transpose so that the stuff seeds are back in the correct place!
        print("tau_seeds_preference_change_matrix",tau_seeds_static_weighting_preference_change_matrix)

        multiplier_matrix_static_weighting_attitude_preference_change = calc_multiplier_matrix(tau_seeds_attitude_preference_change, tau_seeds_static_weighting_preference_change)
        multiplier_matrix_static_weighting_no_preference_change = calc_multiplier_matrix(tau_seeds_no_preference_change, tau_seeds_preference_change_matrix)
        print("multiplier_matrix_static_weighting_attitude_preference_change",multiplier_matrix_static_weighting_attitude_preference_change)
        print("multiplier_matrix_static_weighting_no_preference_change",multiplier_matrix_static_weighting_no_preference_change)

        save_object(multiplier_matrix_static_weighting_attitude_preference_change, fileName + "/Data", "multiplier_matrix_static_weighting_attitude_preference_change") 
        save_object(multiplier_matrix_static_weighting_no_preference_change, fileName + "/Data", "multiplier_matrix_static_weighting_no_preference_change") 

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_mu_target.json",
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_mu_target.json",
        reduction_prop = 0.5,
        carbon_price_duration = 1000,
        static_weighting_run = 1
)

