"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
import time
from copy import deepcopy
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object
from package.resources.run import generate_emissions_load, optimizing_tax_to_reach_emissions_target,generate_data


def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        reduction_prop = 0.5,
        target_tolerance = 0.0001,
        carbon_price_duration = 1000,
        burn_in_duration = 300,
        print_simu = 1,
        ) -> str: 

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "tax_type_emissions_target"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    
    ##################################
    #Gen burn in socieity no carbon price

    params["carbon_price_increased"] = 0
    params["ratio_preference_or_consumption_state"] = 0 #WE ASSUME Consumption BASE LEARNING
    params["alpha_change_state"] = "dynamic_culturally_determined_weights"#burn in we do want preference change to stabalize system
    params["burn_in_duration"] = burn_in_duration
    params["carbon_price_duration"] = 0

    burn_in_run = generate_data(params)

    print("Society burn in done")

    #########################################
    #Now run the burn in simulation with no carbon price for the carbon price time to establish the baseline
    #mu = 0, so preference change but no tax, CONSUMPTION BASED LEARNING

    model_copy_no_tax = deepcopy(burn_in_run)
    model_copy_no_tax.carbon_price_duration = carbon_price_duration#set the carbon price duration, with no burn in period
    model_copy_no_tax.carbon_price_increased = 0#need to run as if the carbon price was zero

    emissions_stock_no_tax = generate_emissions_load(model_copy_no_tax)
    print("emissions_stock_no_tax and guess",emissions_stock_no_tax,carbon_price_duration/2 )
    quit()

    emissions_target = emissions_stock_no_tax*reduction_prop
    stock_target_convergence = emissions_target*target_tolerance
    print("emisisons_target and tolerance", emissions_target,target_tolerance, stock_target_convergence)
    
    #Below also assume that the inital ratio between low and high carbon goods is 1, and both are 1?
    price_ratio = 1
    tau_guess = (1/reduction_prop) - price_ratio#ITS PROPORTIONAL TO THE SIZE OF THE REDUCTION IN EMISSIONS AND THE PRICE ratio of high and low carbon goods()
    #tau_guess = 0.1#
    #print("emissions_stock_target",emissions_target)
    
    ##################################
    #Run for FLAT TAX

    model_copy_flat_tax = deepcopy(burn_in_run)
    model_copy_flat_tax.carbon_tax_implementation =  "flat"
    model_copy_flat_tax.carbon_price_duration = carbon_price_duration#set the carbon price duration, with no burn in period
    model_copy_flat_tax.emissions_stock_target = emissions_target#set the emissions target reduction
    
    best_model_flat_tax , tau_val_flat_tax = optimizing_tax_to_reach_emissions_target(model_copy_flat_tax,tau_guess,stock_target_convergence)
    print("tau_val_flat_tax", tau_val_flat_tax)

    ##################################
    #Run for LINEAR TAX

    model_copy_linear_tax = deepcopy(burn_in_run)
    model_copy_linear_tax.carbon_tax_implementation =  "linear"
    model_copy_linear_tax.carbon_price_duration = carbon_price_duration#set the carbon price duration, with no burn in period
    model_copy_linear_tax.calc_gradient()
    model_copy_linear_tax.emissions_stock_target = emissions_target#set the emissions target reduction
    
    best_model_linear_tax , tau_val_linear_tax = optimizing_tax_to_reach_emissions_target(model_copy_linear_tax,tau_guess,stock_target_convergence)
    print("tau_val_linear_tax", tau_val_linear_tax)


    
    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    

    ##################################
    #save data

    createFolder(fileName)

    save_object(best_model_flat_tax, fileName + "/Data", "best_model_flat_tax")
    save_object(best_model_linear_tax, fileName + "/Data", "best_model_linear_tax")
    save_object(params, fileName + "/Data", "base_params")
    save_object(reduction_prop, fileName + "/Data", "reduction_prop")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_carbon_tax_implementation.json",
        reduction_prop = 0.5,
        target_tolerance = 0.0001,
        carbon_price_duration = 2000,
        burn_in_duration = 300,
)

