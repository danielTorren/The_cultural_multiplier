"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
import numpy as np
from package.resources.utility import ( 
    load_object,
)
from package.resources.plot import (
    plot_identity_timeseries,
    plot_value_timeseries,
    plot_attitude_timeseries,
    plot_total_carbon_emissions_timeseries,
    plot_weighting_matrix_convergence_timeseries,
    plot_cultural_range_timeseries,
    plot_average_identity_timeseries,
    plot_joint_cluster_micro,
    print_live_initial_identity_network,
    live_animate_identity_network_weighting_matrix,
    plot_low_carbon_preferences_timeseries,
    plot_total_flow_carbon_emissions_timeseries
)


def plot_consumption_no_burn_in(fileName, data, dpi_save):

    y_title = r"Quantity"

    fig, axes = plt.subplots(nrows=2,ncols=2, constrained_layout=True)
    inset_ax = axes[1][0].inset_axes([0.4, 0.4, 0.55, 0.55])

    for v in range(data.N):
        data_indivdiual = data.agent_list[v]
        
        axes[0][0].plot(np.asarray(data.history_time),data_indivdiual.history_H_1)
        axes[0][1].plot(np.asarray(data.history_time),data_indivdiual.history_H_2)
        axes[1][0].plot(np.asarray(data.history_time),data_indivdiual.history_L_1)
        axes[1][1].plot(np.asarray(data.history_time),data_indivdiual.history_L_2)
        inset_ax.plot(data.history_time[0:20],data_indivdiual.history_L_1[0:20])
     # [x, y, width, height], x, y are norm 1
    #print("yo",data.history_time[0:20],data_indivdiual.history_L_1[0:20] )
    
    #inset_ax.set_ylim(0,1)
    axes[1][0].set_ylim(0,0.1)
    axes[1][1].set_ylim(0,0.008)

    axes[0][0].set_title("H_1")
    axes[0][1].set_title("H_2")
    axes[1][0].set_title("L_1")
    axes[1][1].set_title("L_2")


    fig.suptitle(r"Service 1 is the luxury and 2 is the basic good, a = 0.8")
    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/quantity_timeseries_preference_no_burn"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_consumption(fileName, data, dpi_save):

    y_title = r"Quantity"

    fig, axes = plt.subplots(nrows=2,ncols=2, constrained_layout=True)

    for v in range(data.N):
        data_indivdiual = data.agent_list[v]
        
        axes[0][0].plot(np.asarray(data.history_time),data_indivdiual.history_H_1)
        axes[0][1].plot(np.asarray(data.history_time),data_indivdiual.history_H_2)
        axes[1][0].plot(np.asarray(data.history_time),data_indivdiual.history_L_1)
        axes[1][1].plot(np.asarray(data.history_time),data_indivdiual.history_L_2)

    axes[0][0].set_title("H_1")
    axes[0][1].set_title("H_2")
    axes[1][0].set_title("L_1")
    axes[1][1].set_title("L_2")


    fig.suptitle(r"Service 1 is the luxury and 2 is the basic good, a = 0.8")
    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/quantity_timeseries_preference"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_emissions_individuals(fileName, data, dpi_save):

    y_title = r"Individuals' emissions flow"

    fig, ax = plt.subplots(constrained_layout=True)

    for v in range(data.N):
        data_indivdiual = data.agent_list[v]
        
        ax.plot(data.history_time,data_indivdiual.history_flow_carbon_emissions)

    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    plotName = fileName + "/Prints"

    f = plotName + "/indi_emisisons_flow_timeseries_preference"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_low_carbon_adoption_timeseries(fileName, data, emissions_threshold_list, dpi_save):

    y_title = r"Low carbon emissions proportion"

    fig, ax = plt.subplots(constrained_layout=True)

    data_matrix = []
    for v in range(data.N):
        data_matrix.append(data.agent_list[v].history_flow_carbon_emissions)

    N = data.N

    data_matrix_trans = np.asarray(data_matrix).T

    for emissions_threshold  in emissions_threshold_list:
        adoption_time_series = [sum([1 if x < emissions_threshold else 0 for x in j])/N for j in data_matrix_trans]
        #print("adoption_time_series",adoption_time_series)
        ax.plot(data.history_time,adoption_time_series, label = r"$E_{Thres} = $" + str(emissions_threshold))

    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    ax.legend()
    plotName = fileName + "/Prints"

    f = plotName + "/plot_low_carbon_adoption_timeserieds" + str(len(emissions_threshold_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_low_carbon_adoption_transition_timeseries(fileName, data,emissions_threshold_range, dpi_save):

    y_title = r"Low carbon emissions proportion"

    fig, ax = plt.subplots(constrained_layout=True)

    data_matrix = []
    for v in range(data.N):
        data_matrix.append(data.agent_list[v].history_flow_carbon_emissions)

    N = data.N

    data_matrix_trans = np.asarray(data_matrix).T

    lowest_threshold = None
    # Iterate through possible thresholds
    for emissions_threshold in emissions_threshold_range:  # Adjust the range as needed
        # Calculate the proportion of agents above the current threshold at the last timestep
        last_timestep_proportion = np.mean(data_matrix_trans[-1, :] < emissions_threshold)

        # Check if the proportion is 1 (all agents are below the threshold)
        if last_timestep_proportion == 1:
            lowest_threshold = emissions_threshold
            break  # Stop iterating if the condition is met

    # Check if a threshold was found
    if lowest_threshold is not None:
        print("Lowest emissions threshold:", lowest_threshold)
        
        lower = (lowest_threshold - 0.000001)
        adoption_time_series = [sum([1 if x < lowest_threshold else 0 for x in j])/N for j in data_matrix_trans]
        ax.plot(data.history_time,adoption_time_series, label = r"$E_{Thres} = $" + str(lowest_threshold))
        adoption_time_series = [sum([1 if x < (lower) else 0 for x in j])/N for j in data_matrix_trans]
        ax.plot(data.history_time,adoption_time_series, label = r"$E_{Thres} = $" + str(lower))
    else:
        print("No threshold found that meets the condition.")



    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    ax.legend()
    plotName = fileName + "/Prints"

    f = plotName + "/plot_low_carbon_adoption_timeserieds"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    PLOT_NAME = "INDIVIDUAL",
    node_size = 50,
    fps = 5,
    interval = 50,
    layout = "circular",
    round_dec = 3,
    dpi_save = 2000,
    no_samples = 10000,
    bandwidth = 0.01,
    shuffle_colours = True,
    animation_save_bool = 0,
    latex_bool = 0
    ) -> None: 

    cmap_multi = get_cmap("plasma")
    cmap_weighting = get_cmap("Reds")

    norm_zero_one = Normalize(vmin=0, vmax=1)
    cmap = LinearSegmentedColormap.from_list(
        "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
    )

    Data = load_object(fileName + "/Data", "social_network")

    ###PLOTS
    """
    if Data.burn_in_duration == 0:
        plot_consumption_no_burn_in(fileName, Data, dpi_save)
    else:
        plot_consumption(fileName, Data, dpi_save)
    """
    #plot_low_carbon_preferences_timeseries(fileName, Data, dpi_save)
    #plot_emissions_individuals(fileName, Data, dpi_save)
    #plot_identity_timeseries(fileName, Data, dpi_save)
    #plot_total_carbon_emissions_timeseries(fileName, Data, dpi_save,latex_bool = latex_bool)
    #plot_total_flow_carbon_emissions_timeseries(fileName, Data, dpi_save,latex_bool = latex_bool)
    threshold_list = [0.0001,0.0002,0.0005,0.001,0.002,0.003,0.004]
    emissions_threshold_range = np.arange(0,0.005,0.000001)
    #plot_low_carbon_adoption_timeseries(fileName, Data,threshold_list, dpi_save)
    plot_low_carbon_adoption_transition_timeseries(fileName, Data,emissions_threshold_range, dpi_save)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/single_experiment_13_57_18__29_09_2023"
    )


