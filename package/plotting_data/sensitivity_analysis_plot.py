"""Performs sobol sensitivity analysis on the model. 

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from SALib.analyze import sobol
import numpy.typing as npt
from package.resources.utility import (
    load_object,
)
from package.resources.plot import (
    multi_scatter_seperate_total_sensitivity_analysis_plot,
)

def get_plot_data(
    problem: dict,
    Y_emissions_stock: npt.NDArray,
    Y_emissions_flow: npt.NDArray,
    Y_mu: npt.NDArray,
    Y_var: npt.NDArray,
    Y_coefficient_of_variance: npt.NDArray,
    Y_emissions_change: npt.NDArray,
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
    Y_mu: npt.NDArray
         values for the mean Individual identity normalized by N*M ie mu/(N*M) at the end of the simulation run time.
         One entry for each parameter set tested
    Y_var: npt.NDArray
         values for the variance of Individual identity in the network at the end of the simulation run time. One entry
         for each parameter set tested
    Y_coefficient_of_variance: npt.NDArray
         values for the coefficient of variance of Individual identity normalized by N*M ie (sigma/mu)*(N*M) in the network
         at the end of the simulation run time. One entry for each parameter set tested
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

    Si_emissions_stock,Si_emissions_flow , Si_mu , Si_var , Si_coefficient_of_variance, Si_emissions_change = analyze_results(problem,Y_emissions_stock,Y_emissions_flow ,Y_mu,Y_var,Y_coefficient_of_variance,Y_emissions_change,calc_second_order) 

    total_emissions_stock, first_emissions_stock = Si_emissions_stock.to_df()
    total_emissions_flow, first_emissions_flow = Si_emissions_flow.to_df()
    total_mu, first_mu = Si_mu.to_df()
    total_var, first_var = Si_var.to_df()
    (
        total_coefficient_of_variance,
        first_coefficient_of_variance,
    ) = Si_coefficient_of_variance.to_df()
    total_emissions_change, first_emissions_change = Si_emissions_change.to_df()

    total_data_sa_emissions_stock, total_yerr_emissions_stock = get_data_bar_chart(total_emissions_stock)
    total_data_sa_emissions_flow, total_yerr_emissions_flow = get_data_bar_chart(total_emissions_flow)
    total_data_sa_mu, total_yerr_mu = get_data_bar_chart(total_mu)
    total_data_sa_var, total_yerr_var = get_data_bar_chart(total_var)
    (
        total_data_sa_coefficient_of_variance,
        total_yerr_coefficient_of_variance,
    ) = get_data_bar_chart(total_coefficient_of_variance)
    total_data_sa_emissions_change, total_yerr_emissions_change = get_data_bar_chart(total_emissions_change)

    first_data_sa_emissions_stock, first_yerr_emissions_stock= get_data_bar_chart(first_emissions_stock)
    first_data_sa_emissions_flow, first_yerr_emissions_flow = get_data_bar_chart(first_emissions_flow)
    first_data_sa_mu, first_yerr_mu = get_data_bar_chart(first_mu)
    first_data_sa_var, first_yerr_var = get_data_bar_chart(first_var)
    (
        first_data_sa_coefficient_of_variance,
        first_yerr_coefficient_of_variance,
    ) = get_data_bar_chart(first_coefficient_of_variance)
    first_data_sa_emissions_change, first_yerr_emissions_change = get_data_bar_chart(first_emissions_change)

    data_sa_dict_total = {
        "emissions_stock": {
            "data": total_data_sa_emissions_stock,
            "yerr": total_yerr_emissions_stock,
        },
        "emissions_flow": {
            "data": total_data_sa_emissions_flow,
            "yerr": total_yerr_emissions_flow,
        },
        "mu": {
            "data": total_data_sa_mu,
            "yerr": total_yerr_mu,
        },
        "var": {
            "data": total_data_sa_var,
            "yerr": total_yerr_var,
        },
        "coefficient_of_variance": {
            "data": total_data_sa_coefficient_of_variance,
            "yerr": total_yerr_coefficient_of_variance,
        },
        "emissions_change": {
            "data": total_data_sa_emissions_change,
            "yerr": total_yerr_emissions_change,
        },
    }
    data_sa_dict_first = {
        "emissions_stock": {
            "data": first_data_sa_emissions_stock,
            "yerr": first_yerr_emissions_stock,
        },
        "emissions_flow": {
            "data": first_data_sa_emissions_flow,
            "yerr": first_yerr_emissions_flow,
        },
        "mu": {
            "data": first_data_sa_mu,
            "yerr": first_yerr_mu,
        },
        "var": {
            "data": first_data_sa_var,
            "yerr": first_yerr_var,
        },
        "coefficient_of_variance": {
            "data": first_data_sa_coefficient_of_variance,
            "yerr": first_yerr_coefficient_of_variance,
        },
        "emissions_change": {
            "data": first_data_sa_emissions_change,
            "yerr": first_yerr_emissions_change,
        },
    }

    return data_sa_dict_total, data_sa_dict_first

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
    #print("data_sa_dict",data_sa_dict)
    #print("plot_dict",plot_dict)
    for i in data_sa_dict.keys():
        for v in plot_dict[i].keys():
            #if v in data_sa_dict:
            data_sa_dict[i][v] = plot_dict[i][v]
            #else:
            #    pass
    return data_sa_dict

def analyze_results(
    problem: dict,
    Y_emissions_stock: npt.NDArray,
    Y_emissions_flow: npt.NDArray,
    Y_mu: npt.NDArray,
    Y_var: npt.NDArray,
    Y_coefficient_of_variance: npt.NDArray,
    Y_emissions_change: npt.NDArray,
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
    
    Si_emissions_flow = sobol.analyze(
        problem,
        Y_emissions_flow,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )

    Si_mu = sobol.analyze(
        problem, Y_mu, calc_second_order=calc_second_order, print_to_console=False
    )
    Si_var = sobol.analyze(
        problem, Y_var, calc_second_order=calc_second_order, print_to_console=False
    )
    Si_coefficient_of_variance = sobol.analyze(
        problem,
        Y_coefficient_of_variance,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )
    Si_emissions_change = sobol.analyze(
        problem,
        Y_emissions_change,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )

    return Si_emissions_stock ,Si_emissions_flow, Si_mu , Si_var , Si_coefficient_of_variance,Si_emissions_change

def main(
    fileName = "results\SA_AV_reps_5_samples_15360_D_vars_13_N_samples_1024",
    plot_outputs = ['emissions_stock','emissions_flow','var',"emissions_change"],
    plot_dict= {
        "emissions_stock": {"title": r"$E/NM$", "colour": "red", "linestyle": "--"},
        "emissions_flow": {"title": r"$E_t/NM$", "colour": "black", "linestyle": ":"},
        "mu": {"title": r"$\mu_{I}$", "colour": "blue", "linestyle": "-"},
        "var": {"title": r"$\sigma^{2}_{I}$", "colour": "green", "linestyle": "*"},
        "coefficient_of_variance": {"title": r"$\sigma/\mu$","colour": "orange","linestyle": "-.",},
        "emissions_change": {"title": r"$\Delta E/NM$", "colour": "brown", "linestyle": "-*"},
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


    Y_emissions_stock = load_object(fileName + "/Data", "Y_emissions_stock")
    #print("emissions_stock",Y_emissions_stock)
    Y_emissions_flow = load_object(fileName + "/Data", "Y_emissions_flow")
    Y_mu = load_object(fileName + "/Data", "Y_mu")
    Y_var = load_object(fileName + "/Data", "Y_var")
    Y_coefficient_of_variance = load_object(fileName + "/Data", "Y_coefficient_of_variance")
    Y_emissions_change = load_object(fileName + "/Data", "Y_emissions_change")

    N_samples = load_object(fileName + "/Data","N_samples" )
    calc_second_order = load_object(fileName + "/Data", "calc_second_order")

    data_sa_dict_total, data_sa_dict_first = get_plot_data(problem, Y_emissions_stock,Y_emissions_flow, Y_mu, Y_var, Y_coefficient_of_variance,Y_emissions_change, calc_second_order)


    data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)
    #print(data_sa_dict_first)
    ###############################

    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first,plot_outputs, titles, N_samples, "First", latex_bool = latex_bool)

    plt.show()
if __name__ == '__main__':

    plots = main(
        fileName="results/sensitivity_analysis_13_13_24__25_04_2023",
        plot_outputs = ['emissions_stock'],#,'emissions_flow','var',"emissions_change"
        plot_dict = {
            "emissions_stock": {"title": r"$E/NM$", "colour": "red", "linestyle": "--"},
            "emissions_flow": {"title": r"$E_t/NM$", "colour": "black", "linestyle": ":"},
            "mu": {"title": r"$\mu_{I}$", "colour": "blue", "linestyle": "-"},
            "var": {"title": r"$\sigma^{2}_{I}$", "colour": "green", "linestyle": "*"},
            "coefficient_of_variance": {"title": r"$\sigma/\mu$","colour": "orange","linestyle": "-.",},
            "emissions_change": {"title": r"$\Delta E/NM$", "colour": "brown", "linestyle": "-*"},
        },
        titles = [
        r"sector substitutability $\nu$",
        r"Initial low carbon preference Beta $b_A$",
        r"sector preference Beta $b_{a}$",
        r"Low carbon substitutability Beta $b_{\sigma}$",
        r"High carbon goods prices Beta $b_{P_H}$"
        ]
    )


