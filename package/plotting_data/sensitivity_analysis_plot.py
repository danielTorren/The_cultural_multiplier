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
        #print("second_emissions_stock", second_emissions_stock, second_emissions_stock.shape)
        #quit()
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

def convert_second_order_data(data, names):


    # Assuming `data` is your DataFrame
    data.reset_index(inplace=True)
    data.rename(columns={"index": "variable_pair"}, inplace=True)

    #new col with names post
    data['variable_pair_indices'] = data['variable_pair'].apply(lambda x: [names.index(param) for param in x])
    
    # Step 1: Create empty numpy matrices
    num_params = len(names)
    interaction_matrix_s2 = np.full([num_params, num_params], np.nan)#np.zeros((num_params, num_params))
    interaction_matrix_s2_conf = np.full([num_params, num_params], np.nan)#np.zeros((num_params, num_params))

    # Step 2: Update matrices with the data from DataFrame
    for index_row, pair_indices in data['variable_pair_indices'].items():
        i, j = pair_indices
        interaction_s2 = data.loc[index_row, 'S2']
        interaction_s2_conf = data.loc[index_row, 'S2_conf']
        interaction_matrix_s2[i, j] = interaction_s2
        interaction_matrix_s2_conf[i, j] = interaction_s2_conf

    return {"S2": interaction_matrix_s2, "S2_conf":interaction_matrix_s2_conf}
    #ileName, data_list_first, plot_outputs, titles, N_samples, "First", network_titles ,latex_bool = latex_boo

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
                #label=data_dict[dict_list[0]]["title"],
            )
        else:
            #print("data_dict", data_dict[dict_list[0]]["data"])
            ax.errorbar(
                data_dict[dict_list[0]]["data"]["ST"].tolist(),
                names,
                xerr=data_dict[dict_list[0]]["yerr"]["ST"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[0]]["colour"],
                #label=data_dict[dict_list[0]]["title"],
            )
        ax.set_title(network_type_list[i])
        ax.set_xlim(left=0)

    fig.supxlabel(r"%s order Sobol index" % (order))
    #axes[2].legend()
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


def multi_scatter_seperate_total_sensitivity_analysis_plot_triple(
    fileName, data_list, dict_list, names, N_samples, order, network_type_list, latex_bool=False
):
    """
    Create scatter chart of results with only the data in `names`.
    """

    fig, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True, sharey=True, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        data_dict = data_list[i]
        
        # Filter data to include only the parameters in `names`
        filtered_data = {}
        filtered_yerr = {}
        y_labels = []
        for key, title in names.items():
            if key in data_dict[dict_list[0]]["data"][order].keys():  # assuming `order` is 'S1' or 'ST'
                filtered_data[key] = data_dict[dict_list[0]]["data"][order][key]
                filtered_yerr[key] = data_dict[dict_list[0]]["yerr"][order][key]
                y_labels.append(title)
        
        # Convert filtered data to lists for plotting
        data_values = list(filtered_data.values())
        yerr_values = list(filtered_yerr.values())
        
        # Plot the filtered data
        ax.errorbar(
            data_values,
            y_labels,
            xerr=yerr_values,
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


def plot_second_order_matrix(fileName, data_list, names, N_samples, order, network_type_list):
    """
    Create second-order sensitivity matrix plot.
    """
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, constrained_layout=True, figsize=(20, 10), sharex=True, sharey = True)#,#sharex=True# figsize=(14, 7) # len(list(data_dict.keys())))
    

    for i in range(ncols):
        data = data_list[i]
        # Extracting data
        second_order_matrix = data["S2"]
        second_order_conf = data["S2_conf"]
        num_vars = len(names)

        # Creating heatmap
        #print(second_order_matrix)
        im = axes[0][i].imshow(second_order_matrix, cmap='viridis')
        cbar = axes[0][i].figure.colorbar(im, ax=axes[0][i])
        if i == 2:
            cbar.set_label('Second-order sensitivity')

        # Creating heatmap
        im_conf = axes[1][i].imshow(second_order_conf, cmap='viridis')
        cbar_conf = axes[1][i].figure.colorbar(im_conf, ax=axes[1][i])
        if i == 2:
            cbar_conf.set_label('Second-order confidence interval')

        # Set ticks and labels
        axes[0][i].set_title(network_type_list[i])

        axes[1][i].set_xticks([])
        #axes[1][i].set_xticks(np.arange(num_vars))
        #axes[1][i].set_xticklabels(names,fontsize=4, rotation=45)

        #fig.supxlabel('Second-order sensitivity matrix')#, fontsize=16

    axes[0][0].set_yticks(np.arange(num_vars))
    axes[0][0].set_yticklabels(names)

    #plt.rc('ytick', labelsize=4) 
    #plt.rc('xtick', labelsize=4) 

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
    #fig.savefig(f, dpi=600, format="eps")
    fig.savefig(f_png, dpi=600, format="png")

def replace_nan_with_neighbors_avg(arr):
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            # Get the left and right neighbors
            left = arr[i - 1] if i - 1 >= 0 else np.nan
            right = arr[i + 1] if i + 1 < len(arr) else np.nan
            
            # Replace NaN with the average of left and right neighbors
            if np.isnan(left):
                arr[i] = right
            elif np.isnan(right):
                arr[i] = left
            else:
                arr[i] = (left + right) / 2
    return arr

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
    #print("problem",problem,len(problem["names"]) - len(titles))
    #quit()

    Y_emissions_stock_SW = load_object(fileName + "/Data", "Y_emissions_stock_SW")
    Y_emissions_stock_SBM = load_object(fileName + "/Data", "Y_emissions_stock_SBM")
    Y_emissions_stock_BA = load_object(fileName + "/Data", "Y_emissions_stock_BA")


    #num_nans = np.isnan( Y_emissions_stock_SW).sum()

    #print("Number of NaNs in the array:", len(Y_emissions_stock_SW), num_nans, (num_nans/len(Y_emissions_stock_SW))*100 )
    #quit()
    """
    Y_emissions_stock_SW = replace_nan_with_neighbors_avg(Y_emissions_stock_SW)
    Y_emissions_stock_SBM = replace_nan_with_neighbors_avg(Y_emissions_stock_SBM)
    Y_emissions_stock_BA = replace_nan_with_neighbors_avg(Y_emissions_stock_BA)
    """

    #"""
    Y_emissions_stock_SW = replace_nan_with_interpolation(Y_emissions_stock_SW)
    Y_emissions_stock_SBM = replace_nan_with_interpolation(Y_emissions_stock_SBM)
    Y_emissions_stock_BA = replace_nan_with_interpolation(Y_emissions_stock_BA)
    #"""
    
    #print(len(Y_emissions_stock_SW))
    #num_nans = np.isnan(Y_emissions_stock_SW).sum()

    #print("Number of NaN values:", num_nans)
    #quit()
    N_samples = load_object(fileName + "/Data","N_samples" )
    calc_second_order = load_object(fileName + "/Data", "calc_second_order")
    #calc_second_order = False

    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")

    if calc_second_order:
        data_sa_dict_total_SW, data_sa_dict_first_SW, second_emissions_stock_df_SW  = get_plot_data(problem, Y_emissions_stock_SW,calc_second_order)  
        data_sa_dict_total_SBM, data_sa_dict_first_SBM, second_emissions_stock_df_SBM  = get_plot_data(problem, Y_emissions_stock_SBM,calc_second_order)  
        data_sa_dict_total_BA, data_sa_dict_first_BA, second_emissions_stock_df_BA  = get_plot_data(problem, Y_emissions_stock_BA,calc_second_order)  
    else:
        data_sa_dict_total_SW, data_sa_dict_first_SW, _ = get_plot_data(problem, Y_emissions_stock_SW,calc_second_order)
        data_sa_dict_total_SBM, data_sa_dict_first_SBM, _ = get_plot_data(problem, Y_emissions_stock_SBM,calc_second_order)
        data_sa_dict_total_BA, data_sa_dict_first_BA, _ = get_plot_data(problem, Y_emissions_stock_BA,calc_second_order)

    data_sa_dict_first_SW = Merge_dict_SA(data_sa_dict_first_SW, plot_dict)
    data_sa_dict_total_SW = Merge_dict_SA(data_sa_dict_total_SW, plot_dict)
    #print("data_sa_dict_total_SW",data_sa_dict_total_SW)
    #print("data_sa_dict_first_SW",)
    #quit()

    data_sa_dict_first_SBM = Merge_dict_SA(data_sa_dict_first_SBM, plot_dict)
    data_sa_dict_total_SBM = Merge_dict_SA(data_sa_dict_total_SBM, plot_dict)

    data_sa_dict_first_BA = Merge_dict_SA(data_sa_dict_first_BA, plot_dict)
    data_sa_dict_total_BA = Merge_dict_SA(data_sa_dict_total_BA, plot_dict)

    #quit()

    ###############################
    #print("titles", len(titles))
    #multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first_SW, plot_outputs, titles, N_samples, "First", "SW", latex_bool = latex_bool)
    #multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first_SBM, plot_outputs, titles, N_samples, "First", "SBM" ,latex_bool = latex_bool)
    #multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first_BA, plot_outputs, titles, N_samples, "First", "BA" ,latex_bool = latex_bool)

    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    data_list_first = [data_sa_dict_first_SW,  data_sa_dict_first_SBM, data_sa_dict_first_BA]
    data_list_total = [data_sa_dict_total_SW,  data_sa_dict_total_SBM, data_sa_dict_total_BA]

    #print(data_sa_dict_first_SW)
    #quit()
    multi_scatter_seperate_total_sensitivity_analysis_plot_triple(fileName, data_list_first, plot_outputs, titles, N_samples, "First", network_titles ,latex_bool = latex_bool)
    multi_scatter_seperate_total_sensitivity_analysis_plot_triple(fileName, data_list_total, plot_outputs, titles, N_samples, "Total", network_titles ,latex_bool = latex_bool)
    #print(titles)
    #quit()

    """
    if calc_second_order:
        #print("variable_parameters_dict", variable_parameters_dict)

        properties_list = list(variable_parameters_dict.keys())

        data_sa_dict_second_SW = convert_second_order_data(second_emissions_stock_df_SW, properties_list)
        data_sa_dict_second_SBM = convert_second_order_data(second_emissions_stock_df_SBM, properties_list)
        data_sa_dict_second_BA =  convert_second_order_data(second_emissions_stock_df_BA, properties_list)

        data_list_second = [data_sa_dict_second_SW,  data_sa_dict_second_SBM, data_sa_dict_second_BA]
        plot_second_order_matrix(fileName, data_list_second,  titles, N_samples, "second", network_titles)
    """
    #TOTAL I DONT THINK WORKS AT THE MOMENT
    #multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_total, plot_outputs, titles, N_samples, "Total", latex_bool = latex_bool)

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
        fileName="results/sensitivity_analysis_10_56_45__23_08_2024",#sensitivity_analysis_SBM_11_21_11__30_01_2024
        plot_outputs = ['emissions_stock'],#,'emissions_flow','var',"emissions_change"
        plot_dict = {
            "emissions_stock": {"title": r"Cumulative emissions, $E$", "colour": "red", "linestyle": "--"},
        },
        titles = [    
            "Social suseptability, $\\phi$",
            "Carbon price, $\\tau$",
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

    titles_dict = {
            "phi_lower": "Social suseptability, $\\phi$",
            "carbon_price_increased_lower": "Carbon price, $\\tau$",
            "N": "Number of individuals, $N$",
            "M": "Number of sectors, $M$",
            "sector_substitutability": "Sector substitutability, $\\nu$",
            "low_carbon_substitutability_lower": "Low carbon substitutability, $\\sigma_{m}$",
            "confirmation_bias": "Confirmation bias, $\\theta$",
            "homophily_state": "Homophily state, $h$",
            "coherance_state": "Coherance state, $c$",
            "a_preferences": "Initial preference Beta, $a$",
            "b_preferences": "Initial preference Beta, $b$",
        }


