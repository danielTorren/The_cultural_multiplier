"""Performs sobol sensitivity analysis on the model. 

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from SALib.analyze import sobol
import numpy.typing as npt
from package.resources.utility import (
    load_object,
    save_object
)
from package.resources.plot import (
    multi_scatter_seperate_total_sensitivity_analysis_plot,
)
import math

def get_plot_data(
    problem: dict,
    Y_emissions_stock: npt.NDArray,
    calc_second_order: bool,
) -> tuple[dict, dict]:
    """
    Take the input results data from the sensitivity analysis  experiments for the four variables measures and now preform the analysis to give
    the total, first (and second order) sobol index values for each parameter varied. Then get this into a nice format that can easily be plotted
    with error bars.
    Parameters
    ----------
    problem: dict
        Outlines the number of variables to be varied, the names of these variables and the bounds that they take
    Y_emissions: npt.NDArray
        values for the Emissions = total network emissions/(N*M) at the end of the simulation run time. One entry for each
        parameter set tested
    calc_second_order: bool
        Whether or not to conduct second order sobol sensitivity analysis, if set to False then only first and total order results will be
        available. Setting to True increases the total number of runs for the sensitivity analysis but allows for the study of interdependancies
        between parameters
    Returns
    -------
    data_sa_dict_total: dict[dict]
        dictionary containing dictionaries each with data regarding the total order sobol analysis results for each output measure
    data_sa_dict_first: dict[dict]
        dictionary containing dictionaries each with data regarding the first order sobol analysis results for each output measure
    """

    Si_emissions_stock = analyze_results(problem,Y_emissions_stock,calc_second_order) 

    if calc_second_order:
        total_emissions_stock, first_emissions_stock, second_emissions_stock = Si_emissions_stock.to_df()
    else:
        total_emissions_stock, first_emissions_stock = Si_emissions_stock.to_df()


    total_data_sa_emissions_stock, total_yerr_emissions_stock = get_data_bar_chart(total_emissions_stock)
    first_data_sa_emissions_stock, first_yerr_emissions_stock= get_data_bar_chart(first_emissions_stock)

    data_sa_dict_total = {
        "emissions_stock": {
            "data": total_data_sa_emissions_stock,
            "yerr": total_yerr_emissions_stock,
        },
    }
    data_sa_dict_first = {
        "emissions_stock": {
            "data": first_data_sa_emissions_stock,
            "yerr": first_yerr_emissions_stock,
        },
    }

    if calc_second_order:
        return data_sa_dict_total, data_sa_dict_first, second_emissions_stock
    else:
        return data_sa_dict_total, data_sa_dict_first, calc_second_order#return nothing for second order
    
    

def get_data_bar_chart(Si_df):
    """
    Taken from: https://salib.readthedocs.io/en/latest/_modules/SALib/plotting/bar.html
    Reduce the sobol index dataframe down to just the bits I want for easy plotting of sobol index and its error

    Parameters
    ----------
    Si_df: pd.DataFrame,
        Dataframe of sensitivity results.
    Returns
    -------
    Sis: pd.Series
        the value of the index
    confs: pd.Series
        the associated error with index
    """

    # magic string indicating DF columns holding conf bound values
    conf_cols = Si_df.columns.str.contains("_conf")
    confs = Si_df.loc[:, conf_cols]  # select all those that ARE in conf_cols!
    confs.columns = [c.replace("_conf", "") for c in confs.columns]
    Sis = Si_df.loc[:, ~conf_cols]  # select all those that ARENT in conf_cols!

    return Sis, confs

def Merge_dict_SA(data_sa_dict: dict, plot_dict: dict) -> dict:
    """
    Merge the dictionaries used to create the data with the plotting dictionaries for easy of plotting later on so that its drawing from
    just one dictionary. This way I seperate the plotting elements from the data generation allowing easier re-plotting. I think this can be
    done with some form of join but I have not worked out how to so far
    Parameters
    ----------
    data_sa_dict: dict
        Dictionary of dictionaries of data associated with each output measure from the sensitivity analysis for a specific sobol index
    plot_dict: dict
        data structure that contains specifics about how a plot should look for each output measure from the sensitivity analysis

    Returns
    -------
    data_sa_dict: dict
        the joined dictionary of dictionaries
    """
    for i in data_sa_dict.keys():
        for v in plot_dict[i].keys():
            data_sa_dict[i][v] = plot_dict[i][v]
    return data_sa_dict

def analyze_results(
    problem: dict,
    Y_emissions_stock: npt.NDArray,
    calc_second_order: bool,
) -> tuple:
    """
    Perform sobol analysis on simulation results
    """
    
    Si_emissions_stock = sobol.analyze(
        problem,
        Y_emissions_stock,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )
    
    return Si_emissions_stock

def main(
    fileName = "results\SA_AV_reps_5_samples_15360_D_vars_13_N_samples_1024",
    plot_outputs = ['emissions_stock'],
    plot_dict= {
        "emissions_stock": {"title": r"$E/NM$", "colour": "red", "linestyle": "--"},
    },
    titles= [
    r"sector  substitutability $\nu$",
    r"Initial low carbon preference Beta $b_A$",
    r"sector preference Beta $b_{a}$",
    r"Low carbon substitutability Beta $b_{\\sigma}$",
    r"High carbon goods prices Beta $b_{P_H}$"
    ],
    latex_bool = 0
    ) -> None: 

    problem = load_object(fileName + "/Data", "problem")
    print("problem",problem,len(problem["names"]) - len(titles))
    #quit()
    Y_emissions_stock = load_object(fileName + "/Data", "Y_emissions_stock")
    print(" Y_emissions_stock", Y_emissions_stock, len(Y_emissions_stock))
    print(sum(math.isnan(x) for x in Y_emissions_stock)) 
    #quit()

    N_samples = load_object(fileName + "/Data","N_samples" )
    calc_second_order = load_object(fileName + "/Data", "calc_second_order")

    if calc_second_order:
        data_sa_dict_total, data_sa_dict_first, second_emissions_stock_df  = get_plot_data(problem, Y_emissions_stock,calc_second_order)  
    else:
        data_sa_dict_total, data_sa_dict_first, ___ = get_plot_data(problem, Y_emissions_stock,calc_second_order)

    data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)
    data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
    ###############################

    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, plot_outputs, titles, N_samples, "First", latex_bool = latex_bool)
    #TOTAL I DONT THINK WORKS AT THE MOMENT
    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_total, plot_outputs, titles, N_samples, "Total", latex_bool = latex_bool)
    
    """
    array_first = data_sa_dict_first['emissions_stock']["data"]["S1"].to_numpy() 
    array_total = data_sa_dict_total['emissions_stock']["data"]["ST"].to_numpy() 
    diff = array_total- array_first
    print("Difference total - first", diff)
    save_object(diff, fileName + "/Data","diff")
    #a = second_emissions_stock_df.plot()
    #print(a)
    """
    plt.show()

    

if __name__ == '__main__':

    plots = main(
        fileName="results/sensitivity_analysis_BA_11_26_31__30_01_2024",#sensitivity_analysis_SBM_11_21_11__30_01_2024
        plot_outputs = ['emissions_stock'],#,'emissions_flow','var',"emissions_change"
        plot_dict = {
            "emissions_stock": {"title": r"Cumulative emissions, $E$", "colour": "red", "linestyle": "--"},
        },
        titles = [    
            "phi_lower",
            "carbon_price",
            "N",
            "M",
            "sector_substitutability",
            #"low_carbon_substitutability_lower",
            "low_carbon_substitutability_upper",
            "std_low_carbon_preference",
            "std_learning_error",
            "confirmation_bias",
            "homophily_state",
            "BA_nodes"
        ]
    )

    """
    BA: 
            titles = [    
            "phi_lower",
            "carbon_price",
            "N",
            "M",
            "sector_substitutability",
            "low_carbon_substitutability_lower",
            "low_carbon_substitutability_upper",
            "std_low_carbon_preference",
            "std_learning_error",
            "confirmation_bias",
            "homophily_state",
            "BA_nodes"
        ]
    SBM:
        titles = [    
            "phi_lower",
            "carbon_price",
            "N",
            "M",
            "sector_substitutability",
            #"low_carbon_substitutability_lower",
            "low_carbon_substitutability_upper",
            "std_low_carbon_preference",
            "std_learning_error",
            "confirmation_bias",
            "homophily_state",
            "SBM_block_num",
            "SBM_network_density_input_intra_block",
            "SBM_network_density_input_inter_block"
        ]
    SW:
            titles = [    
            "phi_lower",
            "carbon_price",
            "N",
            "M",
            "sector_substitutability",
            "low_carbon_substitutability_lower",
            "low_carbon_substitutability_upper",
            "std_low_carbon_preference",
            "std_learning_error",
            "confirmation_bias",
            "homophily_state",
            "SW_network_density",
            "SW_prob_rewire"
            ]
    """


