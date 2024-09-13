"""Generate data for sensitivity analysis

Created: 10/10/2022
"""

# imports
import json
import numpy as np
from SALib.sample import saltelli
import numpy.typing as npt
from package.resources.utility import (
    createFolder,
    save_object,
    produce_name_datetime
)
from package.resources.run import parallel_run_sa

# modules
def generate_problem(
    variable_parameters_dict: dict[dict],
    N_samples: int,
    AV_reps: int,
    calc_second_order: bool,
) -> tuple[dict, str, npt.NDArray]:
    """
    Generate the saltelli.sample given an input set of base and variable parameters, generate filename and folder. Satelli sample used
    is 'a popular quasi-random low-discrepancy sequence used to generate uniform samples of parameter space.' - see the SALib documentation

    Parameters
    ----------
    variable_parameters_dict: dict[dict]
        These are the parameters to be varied. The data is structured as follows: The key is the name of the property, the value is composed
        of another dictionary which contains itself several properties; "property": the parameter name, "min": the minimum value to be used in
        the sensitivity analysis, "max": the maximum value to be used in the sensitivity analysis, "title": the name to be used when plotting
        sensitivity analysis results. See sa_run for example data structure.
    N_samples: int
        Number of samples taken per parameter, If calc_second_order is False, the Satelli sample give N * (D + 2), (where D is the number of parameter) parameter sets to run the model
        .There are then extra runs per parameter set to account for stochastic variation. If calc_second_order is True, then this is N * (2D + 2) parameter sets.
    AV_reps: int
        number of repetitions performed to average over stochastic effects
    calc_second_order: bool
        Whether or not to conduct second order sobol sensitivity analysis, if set to False then only first and total order results will be
        available. Setting to True increases the total number of runs for the sensitivity analysis but allows for the study of interdependancies
        between parameters
    Returns
    -------
    problem: dict
        Outlines the number of variables to be varied, the names of these variables and the bounds that they take
    fileName: str
        name of file where results may be found
    param_values: npt.NDArray
        the set of parameter values which are tested in the sensitivity analysis, generated using saltelli.sample - see:
        https://salib.readthedocs.io/en/latest/api/SALib.sample.html#SALib.sample.saltelli.sample for more details
    """
    D_vars = len(variable_parameters_dict)
    
    if calc_second_order:
        samples = N_samples * (2*D_vars + 2)
    else:
        samples = N_samples * (D_vars + 2)

    print("samples: ", samples)
    print("Total runs: ",samples*AV_reps)

    names_list = [x["property"] for x in variable_parameters_dict.values()]
    bounds_list = [[x["min"], x["max"]] for x in variable_parameters_dict.values()]
    round_variable_list = [x["property"] for x in variable_parameters_dict.values() if x["round"]]

    problem = {
        "num_vars": D_vars,
        "names": names_list,
        "bounds": bounds_list,
    }

    ########################################

    # GENERATE PARAMETER VALUES
    # FOR SOME REASON SOBOL DOESNT WORK IT SAYS SKIP VALUES MUST BE POSTIVE SO IGNORE THE WARNING.
    param_values = saltelli.sample(
        problem, N_samples, calc_second_order=calc_second_order
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    print("round_variable_list:",round_variable_list)
    for i in round_variable_list:
        index_round = problem["names"].index(i)
        param_values[:,index_round] = np.asarray([int(x) for x in np.round(param_values[:,index_round])])
    return problem, param_values

def stochastic_produce_param_list_SA(
    param_values: npt.NDArray, base_params: dict, variable_parameters_dict: dict[dict]
) -> list:
    """
    Generate the list of dictionaries containing informaton for each experiment. We combine the base_params with the specific variation for
    that experiment from param_values and we just use variable_parameters_dict for the property

    Parameters
    ----------
    param_values: npt.NDArray
        the set of parameter values which are tested in the sensitivity analysis, generated using saltelli.sample - see:
        https://salib.readthedocs.io/en/latest/api/SALib.sample.html#SALib.sample.saltelli.sample for more details
    base_params: dict
        This is the set of base parameters which act as the default if a given variable is not tested in the sensitivity analysis.
        See sa_run for example data structure
    variable_parameters_dict: dict[dict]
        These are the parameters to be varied. The data is structured as follows: The key is the name of the property, the value is composed
        of another dictionary which contains itself several properties; "property": the parameter name, "min": the minimum value to be used in
        the sensitivity analysis, "max": the maximum value to be used in the sensitivity analysis, "title": the name to be used when plotting
        sensitivity analysis results. See sa_run for example data structure.

    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []
    seeds_labels = ["preferences_seed", "network_structure_seed", "shuffle_homophily_seed", "shuffle_coherance_seed"]
    for i, X in enumerate(param_values):
        base_params_copy = (
            base_params.copy()
        )  # copy it as we dont want the changes from one experiment influencing another
        variable_parameters_dict_toList = list(
            variable_parameters_dict.values()
        )  # turn it too a list so we can loop through it as X is just an array not a dict
        for v in range(len(X)):  # loop through the properties to be changed
            base_params_copy[variable_parameters_dict_toList[v]["property"]] = X[
                v
            ]  # replace the base variable value with the new value for that experiment
        for j in range(base_params_copy["seed_reps"]):#copy with stochastic seeds
            for k, label in enumerate(seeds_labels):
                base_params_copy[label] = int(10*k + j + 1)
            #base_params_copy["set_seed"] = int(v+1)
            params_list.append(
                base_params_copy.copy()
            )  
        #params_list.append(base_params_copy)
    return params_list

def main(
        N_samples = 1024,
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
        calc_second_order = False
        ) -> str: 

    # load base params
    f = open(BASE_PARAMS_LOAD)
    base_params = json.load(f)

    # load variable params
    f_variable_parameters = open(VARIABLE_PARAMS_LOAD)
    variable_parameters_dict = json.load(f_variable_parameters)
    f_variable_parameters.close()

    ##AVERAGE RUNS
    AV_reps = base_params["seed_reps"]
    print("Average reps: ", AV_reps)

    problem, param_values = generate_problem(
        variable_parameters_dict, N_samples, AV_reps, calc_second_order
    )   

    #SW
    base_params["network_type"] = "SW"
    params_list_sa_SW = stochastic_produce_param_list_SA(
        param_values, base_params, variable_parameters_dict
    )
    
    #SBM
    base_params["network_type"] = "SBM"
    params_list_sa_SBM = stochastic_produce_param_list_SA(
        param_values, base_params, variable_parameters_dict
    )

    #BA
    base_params["network_type"] = "SF"
    params_list_sa_SF = stochastic_produce_param_list_SA(
        param_values, base_params, variable_parameters_dict
    )

    params_list_sa = params_list_sa_SW + params_list_sa_SBM + params_list_sa_SF

    print("Total runs: ", len(params_list_sa))

    root = "sensitivity_analysis"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    Y_emissions_stock_stochastic = parallel_run_sa(
        params_list_sa
    )

    len_y = int(len(params_list_sa_SW)/AV_reps)

    Y_emissions_stock_reshape = Y_emissions_stock_stochastic.reshape(3,len_y,AV_reps)
    Y_emissions_stock = np.mean(Y_emissions_stock_reshape, axis=2)#AVERAGE OVER THE SEED VARIATIONS


    createFolder(fileName)
    
    save_object(base_params, fileName + "/Data", "base_params")
    #save_object(params_list_sa, fileName + "/Data", "params_list_sa")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    save_object(problem, fileName + "/Data", "problem")
    save_object(Y_emissions_stock[0], fileName + "/Data", "Y_emissions_stock_SW")
    save_object(Y_emissions_stock[1], fileName + "/Data", "Y_emissions_stock_SBM")
    save_object(Y_emissions_stock[2], fileName + "/Data", "Y_emissions_stock_SF")
    save_object(Y_emissions_stock_reshape, fileName + "/Data", "Y_emissions_stock_reshape")
    save_object(N_samples , fileName + "/Data","N_samples")
    save_object(calc_second_order, fileName + "/Data","calc_second_order")

    return fileName

if __name__ == '__main__':
    fileName_Figure_6 = main(
    N_samples = 128,
    BASE_PARAMS_LOAD = "package/constants/base_params_sensitivity.json",
    VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json",
    calc_second_order = True
    )