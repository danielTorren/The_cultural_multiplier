# imports
import time
import json
from package.resources.utility import createFolder, produce_name_datetime, save_object
from package.resources.run import emissions_parallel_run, generate_data


def arrange_scenarios_tax(base_params_tax,scenarios_seed):
    #uniform 
    base_params_tax["ratio_preference_or_consumption_state"] = 0 #WE ASSUME Consumption BASE LEARNING
    base_params_tax["alpha_change_state"] = "uniform_network_weighting"
    base_params_tax["confirmation_bias"] = 0
    base_params_tax["carbon_price_increased_lower"] = 0

    #vary seed
    params_list = []
    for i in scenarios_seed:
        base_params_tax["vary_seed_state"] = i
        for v in range(base_params_tax["seed_reps"]):
            base_params_tax["set_seed"] = int(v+1)
            params_list.append(base_params_tax.copy())  

    return params_list

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params_tau_vary.json",
        print_simu = 1,
        scenarios_seed = ["preferences", "network", "shuffle"],
        ) -> str: 

    scenario_reps = len(scenarios_seed)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)
    #print("params", params)
    root = "init_seed_effect_gen"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if print_simu:
        start_time = time.time()
    
    #Gen params lists
    params["network_type"] = "SW"
    params_list_tax_SW = arrange_scenarios_tax(params,scenarios_seed)
    params["network_type"] = "SBM"
    params_list_tax_SBM = arrange_scenarios_tax(params,scenarios_seed)
    params["network_type"] = "BA"
    params_list_tax_BA = arrange_scenarios_tax(params,scenarios_seed)

    params_list = params_list_tax_SW + params_list_tax_SBM + params_list_tax_BA
    print("Total runs: ",len(params_list))
    
    Data_serial = emissions_parallel_run(params_list)
    data_array = Data_serial.reshape(3,scenario_reps,params["seed_reps"])
    data_SW = data_array[0]
    data_SBM = data_array[1]
    data_BA = data_array[2]

    params["alpha_change_state"] = "fixed_preferences"
    fixed_run = generate_data(params)
    fixed_emisisons = fixed_run.total_carbon_emissions_stock
    print(fixed_emisisons)

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    ##################################
    #save data

    createFolder(fileName)
    save_object(fixed_emisisons, fileName + "/Data", "fixed_emissions")
    save_object(data_SW, fileName + "/Data", "emissions_SW")
    save_object(data_SBM, fileName + "/Data", "emissions_SBM")
    save_object(data_BA, fileName + "/Data", "emissions_BA")
    save_object(params, fileName + "/Data", "base_params")
    save_object(scenarios_seed, fileName + "/Data", "scenarios_seed")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_networks_init_seed.json",
        scenarios_seed = ["preferences", "network"]
    )

