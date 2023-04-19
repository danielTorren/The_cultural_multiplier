"""Plot simulation results
A module that use input data or social network object to produce plots for analysis.
These plots also include animations or phase diagrams.

Created: 10/10/2022
"""

# imports
import string
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize, LinearSegmentedColormap, SymLogNorm
from matplotlib.cm import get_cmap
from typing import Union
from package.model.network import Network
from scipy.stats import beta
import numpy.typing as npt

###########################################################
#Setting fonts and font sizes

def set_latex(
    SMALL_SIZE = 14,
    MEDIUM_SIZE = 18,
    BIGGER_SIZE = 22,
):


    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

#########################################################
def prod_pos(layout_type: str, network: nx.Graph) -> nx.Graph:

    if layout_type == "circular":
        pos_identity_network = nx.circular_layout(network)
    elif layout_type == "spring":
        pos_identity_network = nx.spring_layout(network)
    elif layout_type == "kamada_kawai":
        pos_identity_network = nx.kamada_kawai_layout(network)
    elif layout_type == "planar":
        pos_identity_network = nx.planar_layout(network)
    else:
        raise Exception("Invalid layout given")

    return pos_identity_network

##########################################################
#Plot for the figure in the paper
def plot_discount_factors_delta(
    f: str,
    delta_discount_list: list,
    delta_vals: list,
    time_list: npt.NDArray,
    cultural_inertia: float,
    dpi_save: int,
    latex_bool = False
) -> None:
    """
    Plot several distributions for the truncated discounting factor for different parameter values

    Parameters
    ----------
    f: str
        filename, where plot is saved
    const_delta_discount_list: list[list]
        list of time series data of discount factor for the case where discount parameter delta is constant
    delta_vals: list
        values of delta the discount parameter used in graph
    time_list: npt.NDArray
        time points used
    cultural_inertia: float
        the number of steps into the past that are considered when individuals consider their identity
    dpi_save: int
        the dpi of image saved

    Returns
    -------
    None
    """
    if latex_bool:
        set_latex()

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(delta_vals)):
        ax.plot(
            time_list,
            delta_discount_list[i],
            linestyle="--",
            label=r"$\delta$ = %s" % (delta_vals[i]),
        )  # bodge so that we dont repeat one of the lines

    ax.set_xlabel(r"Time steps into past")
    ax.set_ylabel(r"Discount array, $D_t$")
    ax.set_xticks(np.arange(0, -cultural_inertia, step=-20))
    ax.legend()

    fig.savefig(f, dpi=dpi_save, format="eps")

def live_print_identity_timeseries(
    fileName, Data_list, property_varied, dpi_save,latex_bool = False
):
    if latex_bool:
        set_latex()

    fig, axes = plt.subplots(nrows=1, ncols=len(Data_list),figsize=(10, 6), sharey=True)
    y_title = r"Identity, $I_{t,n}$"

    for i, ax in enumerate(axes.flat):
        for v in Data_list[i].agent_list:
            ax.plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_identity)
            )
            #print("v.history_identity",v.history_identity)
        
        ax.text(0.5, 1.03, string.ascii_uppercase[i], transform=ax.transAxes, size=20, weight='bold')

        ax.set_xlabel(r"Time")
        ax.set_ylim(0, 1)

    axes[0].set_ylabel(r"%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/live_plot_identity_timeseries_%s.eps" % property_varied
    fig.savefig(f, dpi=dpi_save, format="eps")

def bifurcation_plot_identity_or_not(fileName,cluster_pos_matrix_identity,cluster_pos_matrix_no_identity,vals_list, dpi_save,latex_bool = False):
    if latex_bool:
        set_latex()
    fig, axes = plt.subplots(nrows = 1, ncols=2, sharey= True, figsize= (10,6))

    for i in range(len(vals_list)):
        x_identity = [vals_list[i]]*(len(cluster_pos_matrix_identity[i]))
        y_identity = cluster_pos_matrix_identity[i]
        axes[0].plot(x_identity,y_identity, ls="", marker=".", color = "k", linewidth = 0.5)

    
        x_no_identity = [vals_list[i]]*(len(cluster_pos_matrix_no_identity[i]))
        y_no_identity = cluster_pos_matrix_no_identity[i]
        axes[1].plot(x_no_identity,y_no_identity, ls="", marker=".", color = "r", linewidth = 0.5)


    axes[0].set_ylim(0,1)

    axes[0].set_title(r"Inter-behavioural dependance")
    axes[1].set_title(r"Behavioural independance")

    axes[0].set_xlabel(r"Confirmation bias, $\theta$")
    axes[1].set_xlabel(r"Confirmation bias, $\theta$")
    axes[0].set_ylabel(r"Final attitude clusters, $m = 1$")
    
    plotName = fileName + "/Plots"
    f = plotName + "/bifurcation_plot_%s" % (len(vals_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def multi_scatter_seperate_total_sensitivity_analysis_plot(
    fileName, data_dict, dict_list, names, dpi_save, N_samples, order,latex_bool = False
):
    """
    Create scatter chart of results.
    """

    if latex_bool:
        set_latex()

    fig, axes = plt.subplots(ncols=len(dict_list), nrows=1, constrained_layout=True , sharey=True,figsize=(12, 6))#,#sharex=True# figsize=(14, 7) # len(list(data_dict.keys())))
    
    plt.rc('ytick', labelsize=4) 

    for i, ax in enumerate(axes.flat):
        if order == "First":
            ax.errorbar(
                data_dict[dict_list[i]]["data"]["S1"].tolist(),
                names,
                xerr=data_dict[dict_list[i]]["yerr"]["S1"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[i]]["colour"],
                label=data_dict[dict_list[i]]["title"],
            )
        else:
            ax.errorbar(
                data_dict[dict_list[i]]["data"]["ST"].tolist(),
                names,
                xerr=data_dict[dict_list[i]]["yerr"]["ST"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[i]]["colour"],
                label=data_dict[dict_list[i]]["title"],
            )
        ax.legend()
        ax.set_xlim(left=0)

    fig.supxlabel(r"%s order Sobol index" % (order))

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_seperate_sensitivity_analysis_plot.eps"
        % (len(names), N_samples, order)
    )
    f_png = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_seperate_sensitivity_analysis_plot.png"
        % (len(names), N_samples, order)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")
    fig.savefig(f_png, dpi=dpi_save, format="png")

def live_print_identity_timeseries_with_weighting(
    fileName, Data_list, property_varied, title_list, dpi_save, cmap,latex_bool = False
):
    if latex_bool:
        set_latex()
    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(14, 7), constrained_layout=True
    )

    y_title = r"Identity, $I_{t,n}$"

    for i in range(3):
        for v in Data_list[i].agent_list:
            axes[0][i].plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_identity)
            )

        axes[0][i].set_xlabel(r"Time")
        axes[0][i].set_ylabel(r"%s" % y_title)
        axes[0][i].set_title(title_list[i], pad=5)
        axes[0][i].set_ylim(0, 1)

        axes[1][i].matshow(
            Data_list[i].history_weighting_matrix[-1],
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )
        axes[1][i].set_xlabel(r"Individual $k$")
        axes[1][i].set_ylabel(r"Individual $n$")

    # colour bar axes
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)),
        ax=axes[1]#axes.ravel().tolist(),
    ) 
    cbar.set_label(r"Social network weighting, $\alpha_{n,k}$", labelpad= 5)

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/lowres_live_print_identity_timeseries_with_weighting_%s.png"
        % property_varied
    )
    fig.savefig(f, dpi=dpi_save, format="png")

def print_live_initial_identity_networks_and_identity_timeseries(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    norm_zero_one,
    cmap,
    node_size,
    round_dec,
    latex_bool = False
):
    if latex_bool:
        set_latex()
    y_title = r"Identity, $I_{t,n}$"
    fig, axes = plt.subplots(
        nrows=2, ncols=len(Data_list), figsize=(14, 7), constrained_layout=True
    )

    for i in range(len(Data_list)):

        G = nx.from_numpy_array(Data_list[i].history_weighting_matrix[0])
        
        pos_identity_network = prod_pos("circular", G)

        axes[0][i].set_title(
            r"{} = {}".format(property, round(property_list[i], round_dec))
        )

        indiv_culutre_list = [v.history_identity[0] for v in Data_list[i].agent_list]

        colour_adjust = norm_zero_one(indiv_culutre_list)
        ani_step_colours = cmap(colour_adjust)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=axes[1][i],
            pos=pos_identity_network,
            node_size=node_size,
            edgecolors="black",
        )

        #####CULTURE TIME SERIES
        for v in Data_list[i].agent_list:
            axes[0][i].plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_identity)
            )

        axes[0][i].set_xlabel(r"Time")
        axes[0][i].set_ylabel(r"%s" % y_title, labelpad=5)
        axes[0][i].set_ylim(0, 1)

    # colour bar axes
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=axes[1]
    )
    cbar.set_label(r"Initial identity, $I_{0,n}$")

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/%s_print_live_initial_identity_networks_and_identity_timeseries.png"
        % (property)
    )
    fig.savefig(f, dpi=dpi_save)

    f_eps = (
        plotName
        + "/%s_print_live_initial_identity_networks_and_identity_timeseries.eps"
        % (property)
    )
    fig.savefig(f_eps, dpi=dpi_save, format="eps")

def plot_single(data_dict_list,fileName_list, dpi_save, latex_bool = 1):
    if latex_bool:
        set_latex()

    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_ylabel( r"Relative $\%$ change in final emissions, $\% \Delta E_{\tau}$")
    ax.set_xlabel(r"Initial attitude distance between green- and non-influencers")
    mean_list =  data_dict_list[0]["mean_list"]
    mu_emissions_difference_matrix_compare_identity = data_dict_list[0]["emissions_difference_matrix_compare_identity"].mean(axis=1)
    min_emissions_difference_matrix_compare_identity = data_dict_list[0]["emissions_difference_matrix_compare_identity"].min(axis=1)
    max_emissions_difference_matrix_compare_identity = data_dict_list[0]["emissions_difference_matrix_compare_identity"].max(axis=1)

    mu_emissions_difference_matrix_compare_no_identity = data_dict_list[0]["emissions_difference_matrix_compare_no_identity"].mean(axis=1)
    min_emissions_difference_matrix_compare_no_identity = data_dict_list[0]["emissions_difference_matrix_compare_no_identity"].min(axis=1)
    max_emissions_difference_matrix_compare_no_identity = data_dict_list[0]["emissions_difference_matrix_compare_no_identity"].max(axis=1)

    ax.plot(mean_list[::-1],mu_emissions_difference_matrix_compare_identity, ls="-", linewidth = 0.5, color='black', label = r"Inter-behavioural dependence")
    ax.fill_between(mean_list[::-1], min_emissions_difference_matrix_compare_identity, max_emissions_difference_matrix_compare_identity, facecolor='black', alpha=0.5)
    ax.plot(mean_list[::-1],mu_emissions_difference_matrix_compare_no_identity, ls="--", linewidth = 0.5, color='red', label = r"Behavioural independence")
    ax.fill_between(mean_list[::-1], min_emissions_difference_matrix_compare_no_identity, max_emissions_difference_matrix_compare_no_identity, facecolor='red', alpha=0.5)
    ax.legend(loc = "lower right")

    for fileName in fileName_list:
        plotName = fileName + "/Plots"
        f = plotName + "/plot_single_%s" % (len(mean_list))
        fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
        fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_diff_emissions_comparison(data_dict_list,fileName_list, dpi_save, latex_bool = 0):
    if latex_bool:
        set_latex()

    fig, axes = plt.subplots(nrows = 2, ncols = len(data_dict_list), figsize=(14,12), constrained_layout=True, sharey="row", sharex="col")

    axes[0][0].set_ylabel(r"Final emissions, $E_{\tau}$")
    axes[1][0].set_ylabel( r"Relative $\%$ change in final emissions, $\% \Delta E_{\tau}$")

    for i, ax in enumerate(axes[0]):
        mean_list =  data_dict_list[i]["mean_list"]

        mu_emissions_id_array_no_green_no_identity =  data_dict_list[i]["carbon_emissions_not_influencer_no_green_no_identity"].mean(axis=1)
        min_emissions_id_array_no_green_no_identity =  data_dict_list[i]["carbon_emissions_not_influencer_no_green_no_identity"].min(axis=1)
        max_emissions_id_array_no_green_no_identity =  data_dict_list[i]["carbon_emissions_not_influencer_no_green_no_identity"].max(axis=1)

        mu_emissions_id_array_no_green_identity =  data_dict_list[i]["carbon_emissions_not_influencer_no_green_identity"].mean(axis=1)
        min_emissions_id_array_no_green_identity =  data_dict_list[i]["carbon_emissions_not_influencer_no_green_identity"].min(axis=1)
        max_emissions_id_array_no_green_identity =  data_dict_list[i]["carbon_emissions_not_influencer_no_green_identity"].max(axis=1)

        mu_emissions_id_array_green_no_identity =  data_dict_list[i]["carbon_emissions_not_influencer_green_no_identity"].mean(axis=1)
        min_emissions_id_array_green_no_identity =  data_dict_list[i]["carbon_emissions_not_influencer_green_no_identity"].min(axis=1)
        max_emissions_id_array_green_no_identity =  data_dict_list[i]["carbon_emissions_not_influencer_green_no_identity"].max(axis=1)

        mu_emissions_id_array_green_identity =  data_dict_list[i]["carbon_emissions_not_influencer_green_identity"].mean(axis=1)
        min_emissions_id_array_green_identity =  data_dict_list[i]["carbon_emissions_not_influencer_green_identity"].min(axis=1)
        max_emissions_id_array_green_identity =  data_dict_list[i]["carbon_emissions_not_influencer_green_identity"].max(axis=1)

        # cultuer vs no culteur repsresneted black vs red
        # green vs no green by solid vs dashed line
        ax.plot(mean_list[::-1],mu_emissions_id_array_no_green_identity, ls="-", color='black', label = r"Inter-behavioural dependence, No green influencers")
        ax.fill_between(mean_list[::-1], min_emissions_id_array_no_green_identity, max_emissions_id_array_no_green_identity, facecolor='black', alpha=0.5)
        
        ax.plot(mean_list[::-1],mu_emissions_id_array_no_green_no_identity, ls="--", color='red', label = r"Behavioural independence, No green influencers")
        ax.fill_between(mean_list[::-1], min_emissions_id_array_no_green_no_identity, max_emissions_id_array_no_green_no_identity, facecolor='red', alpha=0.5)

        ax.plot(mean_list[::-1],mu_emissions_id_array_green_identity, ls="-", color='green', label = r"Inter-behavioural dependence, Green influencers")
        ax.fill_between(mean_list[::-1], min_emissions_id_array_green_identity, max_emissions_id_array_green_identity, facecolor='green', alpha=0.5)

        ax.plot(mean_list[::-1],mu_emissions_id_array_green_no_identity, ls="--", color='blue', label = r"Behavioural independence, Green influencers")
        ax.fill_between(mean_list[::-1], min_emissions_id_array_green_no_identity, max_emissions_id_array_green_no_identity, facecolor='blue', alpha=0.5)

        
        ax.set_title(r"Confirmation bias, $\theta = %s$" % ( data_dict_list[i]["base_params"]["confirmation_bias"]))

    for i, ax in enumerate(axes[1]):
        mean_list =  data_dict_list[i]["mean_list"]
        mu_emissions_difference_matrix_compare_identity = data_dict_list[i]["emissions_difference_matrix_compare_identity"].mean(axis=1)
        min_emissions_difference_matrix_compare_identity = data_dict_list[i]["emissions_difference_matrix_compare_identity"].min(axis=1)
        max_emissions_difference_matrix_compare_identity = data_dict_list[i]["emissions_difference_matrix_compare_identity"].max(axis=1)

        mu_emissions_difference_matrix_compare_no_identity = data_dict_list[i]["emissions_difference_matrix_compare_no_identity"].mean(axis=1)
        min_emissions_difference_matrix_compare_no_identity = data_dict_list[i]["emissions_difference_matrix_compare_no_identity"].min(axis=1)
        max_emissions_difference_matrix_compare_no_identity = data_dict_list[i]["emissions_difference_matrix_compare_no_identity"].max(axis=1)
    
        ax.plot(mean_list[::-1],mu_emissions_difference_matrix_compare_identity, ls="-", linewidth = 0.5, color='black', label = r"Inter-behavioural dependence")
        ax.fill_between(mean_list[::-1], min_emissions_difference_matrix_compare_identity, max_emissions_difference_matrix_compare_identity, facecolor='black', alpha=0.5)
        ax.plot(mean_list[::-1],mu_emissions_difference_matrix_compare_no_identity, ls="--", linewidth = 0.5, color='red', label = r"Behavioural independence")
        ax.fill_between(mean_list[::-1], min_emissions_difference_matrix_compare_no_identity, max_emissions_difference_matrix_compare_no_identity, facecolor='red', alpha=0.5)
        
        ax.set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")

    axes[0][1].legend(loc = "lower right", fontsize="small")
    axes[1][1].legend(loc = "lower right")

    for fileName in fileName_list:
        plotName = fileName + "/Plots"
        f = plotName + "/plot_diff_emissions_comparison_%s" % (len(mean_list))
        fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
        fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_beta_alt(f:str, a_b_combo_list: list,latex_bool = False ):
    if latex_bool:
        set_latex()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0,1,100)

    for i in a_b_combo_list:
        y = beta.pdf(x, i[0], i[1])
        ax.plot(x,y, label = r"a = %s, b = %s" % (i[0],i[1]))

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"Probability Density Function")
    ax.legend()

    fig.savefig(f + "%s" % (len(a_b_combo_list)) + ".eps", format="eps")

def double_phase_diagram(
    fileName, Z, Y_title, Y_param, variable_parameters_dict, cmap, dpi_save, levels,latex_bool = False
):
    if latex_bool:
        set_latex()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_xlabel(r"Initial attitude Beta, $b_A$")
    ax.set_ylabel(r"Initial attitude Beta, $a_A$")

    X, Y = np.meshgrid(col_dict["vals"], row_dict["vals"])

    cp = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5, levels = levels)
    cbar = fig.colorbar(
        cp,
        ax=ax,
    )
    cbar.set_label(Y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_multirun_double_phase_diagram_%s" % (Y_param)
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")
    
def plot_joint_cluster_micro(fileName, Data, clusters_index_lists,cluster_example_identity_list, vals_time_data, dpi_save, auto_bandwidth, bandwidth,cmap_multi, norm_zero_one,shuffle_colours,latex_bool = False) -> None:
    if latex_bool:
        set_latex()
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6), constrained_layout=True)

    ###################################################

    cmap = get_cmap(name='viridis', lut = len(cluster_example_identity_list))
    ani_step_colours = [cmap(i) for i in range(len(cluster_example_identity_list))] 

    if shuffle_colours:
        np.random.shuffle(ani_step_colours)


    colours_dict = {}#It cant be a list as you need to do it out of order
    for i in range(len(clusters_index_lists)):#i is the list of index in that cluster
        for j in clusters_index_lists[i]:#j is an index in that cluster
            colours_dict["%s" % (j)] = ani_step_colours[i]
        
    for v in range(len(Data.agent_list)):
        axes[0].plot(np.asarray(Data.history_time), np.asarray(Data.agent_list[v].history_identity), color = colours_dict["%s" % (v)])

    axes[0].set_ylabel(r"Identity, $I_{t,n}$")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel(r"Time")

    ##################################################

    inverse_N_g_list = [1/len(i) for i in clusters_index_lists]

    #colour_adjust = norm_zero_one(cluster_example_identity_list)
    #ani_step_colours = cmap(colour_adjust)

    for i in range(len(clusters_index_lists)): 
        axes[1].plot(Data.history_time, vals_time_data[i], color = ani_step_colours[i])#, label = "Cluster %s" % (i + 1)
        axes[1].axhline(y= inverse_N_g_list[i], color = ani_step_colours[i], linestyle = "--")

    #ax.set_title(title_list[z])
    #axes[1].legend()

    axes[1].set_ylabel(r"Mean cluster weighting")
    axes[1].set_xlabel(r"Time")

    #cbar = fig.colorbar(
    #    plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=axes[1]
    #)
    #cbar.set_label(r"Identity, $I_{t,n}$")

    plotName = fileName + "/Prints"
    f = plotName + "/plot_joint_cluster_micro_%s_%s" % (auto_bandwidth, bandwidth)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


##############################################################
#Single shot plots
def plot_identity_timeseries(fileName, Data, dpi_save,latex_bool = False):
    if latex_bool:
        set_latex()
    fig, ax = plt.subplots(figsize=(10,6))
    y_title = r"Identity, $I_{t,n}$"

    for v in Data.agent_list:
        ax.plot(np.asarray(Data.history_time), np.asarray(v.history_identity))
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_ylim(0, 1)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_identity_timeseries"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_individual_timeseries(
    fileName: str,
    Data: Network,
    y_title: str,
    property: str,
    dpi_save: int,
    ylim_low: int,
    latex_bool = False
):
    if latex_bool:
        set_latex()
    fig, axes = plt.subplots(nrows=1, ncols=Data.M, figsize=(14, 7), sharey=True)

    for i, ax in enumerate(axes.flat):
        for v in range(len(Data.agent_list)):
            data_ind = np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property)))
            ax.plot(np.asarray(Data.history_time), data_ind[:, i])

        ax.set_title(r"$\phi_{%s} = %s$" % ((i + 1),  Data.phi_array[i]))
        ax.set_xlabel(r"Time")
        
        ax.set_ylim(ylim_low, 1)

    axes[0].set_ylabel(r"%s" % y_title)
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_%s_timeseries.eps" % property
    fig.savefig(f, dpi=dpi_save, format="eps")

def plot_value_timeseries(fileName: str, Data , dpi_save: int,latex_bool = False):
    if latex_bool:
        set_latex()
    y_title = r"Behavioural value, $B_{t,n,m}$"
    property = "history_behaviour_values"
    ylim_low = -1

    plot_individual_timeseries(fileName, Data, y_title, property, dpi_save, ylim_low)

def plot_attitude_timeseries(fileName: str, Data, dpi_save: int,latex_bool = False):
    if latex_bool:
        set_latex()
    y_title = r"Behavioural attiude, $A_{t,n,m}$"
    property = "history_behaviour_attitudes"
    ylim_low = 0

    plot_individual_timeseries(fileName, Data, y_title, property, dpi_save, ylim_low)

def print_live_initial_identity_network(
    fileName: str,
    Data,
    dpi_save: int,
    layout: str,
    norm_zero_one,
    cmap,
    node_size,
    latex_bool = False
):
    if latex_bool:
        set_latex()
    fig, ax = plt.subplots()
    
    G = nx.from_numpy_array(Data.history_weighting_matrix[0])
    pos_identity_network = prod_pos(layout, G)

    indiv_culutre_list = [v.history_identity[0] for v in Data.agent_list]

    colour_adjust = norm_zero_one(indiv_culutre_list)

    ani_step_colours = cmap(colour_adjust)

    nx.draw(
        G,
        node_color=ani_step_colours,
        ax=ax,
        pos=pos_identity_network,
        node_size=node_size,
        edgecolors="black",
    )

    # colour bar axes
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=ax
    )
    cbar.set_label(r"Identity, $I_{t,n}$")

    plotName = fileName + "/Prints"
    f = plotName + "/print_live_initial_identity_network"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_network_timeseries(
    fileName: str, Data: Network, y_title: str, property: str, dpi_save: int,latex_bool = False
):
    if latex_bool:
        set_latex()
    fig, ax = plt.subplots(figsize=(10,6))
    data = eval("Data.%s" % property)

    # bodge
    ax.plot(Data.history_time, data)
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/" + property + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")

def plot_cultural_range_timeseries(fileName: str, Data, dpi_save: int,latex_bool = False):
    if latex_bool:
        set_latex()
    y_title = "Identity variance"
    property = "history_var_identity"
    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_weighting_matrix_convergence_timeseries(
    fileName: str, Data, dpi_save: int,latex_bool = False
):
    if latex_bool:
        set_latex()
    y_title = "Change in Agent Link Strength"
    property = "history_weighting_matrix_convergence"
    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_total_carbon_emissions_timeseries(
    fileName: str, Data, dpi_save: int,latex_bool = False
):
    if latex_bool:
        set_latex()
    y_title = "Carbon Emissions"
    property = "history_total_carbon_emissions"
    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_average_identity_timeseries(fileName: str, Data, dpi_save: int,latex_bool = False):
    if latex_bool:
        set_latex()
    y_title = "Average identity"
    property = "history_average_identity"

    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)

def live_animate_identity_network_weighting_matrix(
    fileName: str,
    Data: list,
    cmap_weighting: Union[LinearSegmentedColormap, str],
    interval: int,
    fps: int,
    round_dec: int,
    layout: str,
    cmap_identity: Union[LinearSegmentedColormap, str],
    node_size: int,
    norm_zero_one: SymLogNorm,
    save_bool = 0,
    latex_bool = False
):
    if latex_bool:
        set_latex()
        
    def update(i, Data, axes, cmap_identity,cmap_weighting, layout, title):

        #axes[0].clear()
        #axes[1].clear()

        individual_identity_list = [x.identity for x in Data.agent_list]
        colour_adjust = norm_zero_one(individual_identity_list)
        ani_step_colours = cmap_identity(colour_adjust)

        G = nx.from_numpy_array(Data.history_weighting_matrix[i])

        # get pos
        pos = prod_pos(layout, G)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=axes[0],
            pos=pos,
            node_size=node_size,
            edgecolors="black",
        )

        axes[1].matshow(
            Data.history_weighting_matrix[i],
            cmap=cmap_weighting,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )

        axes[1].set_xlabel("Individual $k$")
        axes[1].set_ylabel("Individual $n$")

        title.set_text(
            "Time= {}".format(Data.history_time[i])
        )

    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True)# figsize=(5,6)
    title = fig.suptitle(t="", fontsize=20)

    cbar_identity = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_identity),
        ax=axes[0],
        location="left",
    )  # This does a mapabble on the fly i think, not sure
    cbar_identity.set_label(r"Identity, $I_{t,n}$")

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_weighting),
        ax=axes[1],
        location="right",
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label(r"Social network weighting, $\alpha_{n,k}$")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data.history_time)),
        fargs=(Data, axes, cmap_identity,cmap_weighting, layout, title),
        repeat_delay=500,
        interval=interval,
    )

    if save_bool:
        # save the video
        animateName = fileName + "/Animations"
        f = (
            animateName
            + "/live_animate_identity_network_weighting_matrix.mp4"
        )
        # print("f", f)
        writervideo = animation.FFMpegWriter(fps=fps)
        ani.save(f, writer=writervideo)

    return ani