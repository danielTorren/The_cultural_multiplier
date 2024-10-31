# imports
import matplotlib.pyplot as plt
from SALib.analyze import sobol
import numpy.typing as npt
from package.resources.utility import (
    load_object
)
import numpy as np

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


def multi_scatter_seperate_total_sensitivity_analysis_plot_triple(
    fileName, data_list, dict_list, names, N_samples, order, network_type_list,  latex_bool = False
):
    """
    Create scatter chart of results.
    """

    fig, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True , sharey=True,figsize=(12, 6))#,#sharex=True# figsize=(14, 7) # len(list(data_dict.keys())))
    
    #plt.rc('ytick', labelsize=4) 
    for i, ax in enumerate(axes.flat):
        data_dict = data_list[i]
        if order == "First":
            ax.errorbar(
                data_dict[dict_list[0]]["data"]["S1"].tolist(),
                names,
                xerr=data_dict[dict_list[0]]["yerr"]["S1"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[0]]["colour"],
            )
        else:
            ax.errorbar(
                data_dict[dict_list[0]]["data"]["ST"].tolist(),
                names,
                xerr=data_dict[dict_list[0]]["yerr"]["ST"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[0]]["colour"],
            )
        ax.set_title(network_type_list[i])
        ax.set_xlim(left=0)

    fig.supxlabel(r"%s order Sobol index" % (order))
    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_seperate_sensitivity_analysis_plot_triple.eps"
        % (len(names), N_samples, order)
    )
    f_png = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_seperate_sensitivity_analysis_plot_triple.png"
        % (len(names), N_samples, order)
    )
    fig.savefig(f, dpi=600, format="eps")
    fig.savefig(f_png, dpi=600, format="png")


def replace_nan_with_interpolation(arr):
    # Get the indices where the values are not NaN
    not_nan_indices = np.where(~np.isnan(arr))[0]
    
    # Get the indices where the values are NaN
    nan_indices = np.where(np.isnan(arr))[0]
    
    # Interpolate the NaN indices based on the not NaN values
    arr[nan_indices] = np.interp(nan_indices, not_nan_indices, arr[not_nan_indices])
    
    return arr

def main(
    fileName = "results\SA_AV_reps_5_samples_15360_D_vars_13_N_samples_1024",
    plot_outputs = ['emissions_stock'],
    plot_dict= {
        "emissions_stock": {"title": r"$E/NM$", "colour": "red", "linestyle": "--"},
    },
    titles= ["isnerttitle"],
    latex_bool = 0
    ) -> None: 

    problem = load_object(fileName + "/Data", "problem")

    Y_emissions_stock_SW = load_object(fileName + "/Data", "Y_emissions_stock_SW")
    Y_emissions_stock_SBM = load_object(fileName + "/Data", "Y_emissions_stock_SBM")
    Y_emissions_stock_SF = load_object(fileName + "/Data", "Y_emissions_stock_BA")
    
    print("Amount of Data (Averaged over stochastic runs)", np.count_nonzero(Y_emissions_stock_SW), np.count_nonzero(Y_emissions_stock_SBM), np.count_nonzero(Y_emissions_stock_SF ))
    print("Amount of Nans in Data", np.count_nonzero(np.isnan(Y_emissions_stock_SW)), np.count_nonzero(np.isnan(Y_emissions_stock_SBM)), np.count_nonzero(np.isnan(Y_emissions_stock_SF )))

    #There are sometimes be a few of nans in the data, interpolate the closest values to acount for this as dont want to throw away 150,000 + rusn for a few errors. Usually however this is not required.
    Y_emissions_stock_SW = replace_nan_with_interpolation(Y_emissions_stock_SW)
    Y_emissions_stock_SBM = replace_nan_with_interpolation(Y_emissions_stock_SBM)
    Y_emissions_stock_SF = replace_nan_with_interpolation(Y_emissions_stock_SF)

    N_samples = load_object(fileName + "/Data","N_samples" )
    calc_second_order = load_object(fileName + "/Data", "calc_second_order")

    #variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")

    data_sa_dict_total_SW, data_sa_dict_first_SW, _ = get_plot_data(problem, Y_emissions_stock_SW,calc_second_order)
    data_sa_dict_total_SBM, data_sa_dict_first_SBM, _ = get_plot_data(problem, Y_emissions_stock_SBM,calc_second_order)
    data_sa_dict_total_SF, data_sa_dict_first_SF, _ = get_plot_data(problem, Y_emissions_stock_SF,calc_second_order)

    data_sa_dict_first_SW = Merge_dict_SA(data_sa_dict_first_SW, plot_dict)
    data_sa_dict_total_SW = Merge_dict_SA(data_sa_dict_total_SW, plot_dict)

    data_sa_dict_first_SBM = Merge_dict_SA(data_sa_dict_first_SBM, plot_dict)
    data_sa_dict_total_SBM = Merge_dict_SA(data_sa_dict_total_SBM, plot_dict)

    data_sa_dict_first_SF = Merge_dict_SA(data_sa_dict_first_SF, plot_dict)
    data_sa_dict_total_SF = Merge_dict_SA(data_sa_dict_total_SF, plot_dict)

    network_titles = ["Small-World", "Stochastic Block Model", "Scale-Free"]
    data_list_first = [data_sa_dict_first_SW,  data_sa_dict_first_SBM, data_sa_dict_first_SF]
    data_list_total = [data_sa_dict_total_SW,  data_sa_dict_total_SBM, data_sa_dict_total_SF]

    multi_scatter_seperate_total_sensitivity_analysis_plot_triple(fileName, data_list_first, plot_outputs, titles, N_samples, "First", network_titles ,latex_bool = latex_bool)
    multi_scatter_seperate_total_sensitivity_analysis_plot_triple(fileName, data_list_total, plot_outputs, titles, N_samples, "Total", network_titles ,latex_bool = latex_bool)

    plt.show()


if __name__ == '__main__':

    plots = main(
        fileName="results/sensitivity_analysis_20_46_11__26_08_2024",#sensitivity_analysis_SBM_11_21_11__30_01_2024
        plot_outputs = ['emissions_stock'],#,'emissions_flow','var',"emissions_change"
        plot_dict = {
            "emissions_stock": {"title": r"Cumulative emissions, $E$", "colour": "red", "linestyle": "--"},
        },
        titles = [    
            "Social suseptability, $\\phi$",
            "Carbon tax, $\\tau$",
            "Number of individuals, $N$",
            "Number of sectors, $M$",
            "Sector substitutability, $\\nu$",
            "Low carbon substitutability, $\\sigma_{m}$",
            "Confirmation bias, $\\theta$",
            "Homophily state, $h$",
            "Coherance state, $c$",
            "Initial preference Beta, $a$ ",
            "Initial preference Beta, $b$ ",
        ]
    )

