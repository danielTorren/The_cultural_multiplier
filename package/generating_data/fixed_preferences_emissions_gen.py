# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object


def generate_vals(variable_parameters_dict):
    if variable_parameters_dict["property_divisions"] == "linear":
        property_values_list  = np.linspace(variable_parameters_dict["property_min"], variable_parameters_dict["property_max"], variable_parameters_dict["property_reps"])
    elif variable_parameters_dict["property_divisions"] == "log":
        property_values_list  = np.logspace(np.log10(variable_parameters_dict["property_min"]),np.log10( variable_parameters_dict["property_max"]), variable_parameters_dict["property_reps"])
    else:
        print("Invalid divisions, try linear or log")
    return property_values_list 

def produce_param_list_stochastic_init_seed(params: dict, property_list: list, property: str) -> list[dict]:
    params_list = []
    for i in property_list:
        params[property] = i
        for v in range(params["seed_reps"]):
            params["init_vals_seed"] = int(v+1)
            params_list.append(
                params.copy()
            )  
    return params_list

################################################################################################
###############################################################################################

def gen_A_alt(a_identity,b_identity,std_low_carbon_preference,N,M,clipping_epsilon,init_vals_seed):
    #For inital construction set a seed, this is the same for all runs, then later change it to set_seed
    np.random.seed(init_vals_seed)

    indentities_beta = np.random.beta( a_identity, b_identity, size= N)

    preferences_uncapped = np.asarray([np.random.normal(identity, std_low_carbon_preference, size=M) for identity in  indentities_beta])

    low_carbon_preference_matrix = np.clip(preferences_uncapped, 0 + clipping_epsilon, 1- clipping_epsilon)
    return low_carbon_preference_matrix

def calc_inputs(parameters):
    high_carbon_goods_price = 1

    t_max = parameters["carbon_price_duration"]
    N = int(parameters["N"])
    M = int(parameters["M"])
    a = np.tile(np.asarray([1/M]*M), (N,1))
    if parameters["carbon_tax_implementation"] == "linear":
        PH = np.linspace(high_carbon_goods_price + parameters["init_carbon_price"], high_carbon_goods_price + parameters["carbon_price_increased"],parameters["carbon_price_duration"])
    elif parameters["carbon_tax_implementation"] == "flat":
        PH = np.asarray([high_carbon_goods_price + parameters["carbon_price_increased"]]*parameters["carbon_price_duration"])
    else:
        print("INVALID carbon price implementation")
    nu = parameters["sector_substitutability"]
    A = gen_A_alt(parameters["a_identity"],parameters["b_identity"],parameters["std_low_carbon_preference"],N,M,parameters["clipping_epsilon"],parameters["init_vals_seed"])
    sigma = np.linspace(parameters["low_carbon_substitutability_lower"], parameters["low_carbon_substitutability_upper"], num=M)
    
    return t_max, N, M, a, PH, nu, A, sigma

def calculate_emissions(t_max, N, M, a, P_H, A, sigma, nu):
    emissions = np.zeros(t_max)
    for t in range(t_max):
        numerator = np.zeros((N, M))
        denominator = np.zeros((N, M))
        
        omega = ((P_H[t] * A) / (1 - A))**sigma
        chi = (a / P_H[t])**nu * (
            A * omega**((sigma - 1) / sigma) + (1 - A)
        )**(((nu - 1)*sigma)/(sigma - 1))
        
        numerator = np.sum(chi, axis=1)
        denominator = np.sum(chi * (omega + P_H[t]), axis=1)
        
        emissions[t] = np.sum(numerator / denominator)

    return np.sum(emissions)

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
         ) -> str: 

    f_var = open(VARIABLE_PARAMS_LOAD)
    var_params = json.load(f_var) 

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
    property_reps = var_params["property_reps"]#10,

    property_values_list = generate_vals(
        var_params
    )
    #property_values_list = np.linspace(property_min, property_max, property_reps)

    f = open(BASE_PARAMS_LOAD)
    params = json.load(f)

    root = "fixed_preferences"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    
    createFolder(fileName)

    params_list = produce_param_list_stochastic_init_seed(params, property_values_list, property_varied)
    print("RUNS:",params_list )
    results = []
    for params in params_list:
        #6_11_23
        t_max, N, M, a, P_H, nu, A, sigma = calc_inputs(params)
        res = calculate_emissions(t_max, N, M, a, P_H, A, sigma, nu)  

        results.append(res)
    results_array = np.asarray(results)
    results_reshaped =  results_array.reshape(property_reps,params["seed_reps"])
    
    save_object(params, fileName + "/Data", "base_params")
    save_object(var_params,fileName + "/Data" , "var_params")
    save_object(property_values_list,fileName + "/Data", "property_values_list")
    save_object(results_reshaped,fileName + "/Data", "results")

    return fileName, results_reshaped

if __name__ == '__main__':
    fileName_Figure_1, results_reshaped = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_fixed_preferences_tau.json",#"package/constants/base_params_confirmation_bias.json",#"package/constants/base_params_std_low_carbon_preference.json"
        VARIABLE_PARAMS_LOAD = "package/constants/oneD_dict_tau.json",#"package/constants/oneD_dict_confirmation_bias.json",#"package/constants/oneD_dict_std_low_carbon_preference.json"
)