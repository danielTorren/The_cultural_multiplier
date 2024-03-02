"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap, ScalarMappable
import numpy as np
import matplotlib.markers as mmarkers
from matplotlib.animation import FuncAnimation
from package.resources.utility import ( 
    load_object,
)
import pandas as pd
import seaborn as sns
import networkx as nx
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
    plot_total_flow_carbon_emissions_timeseries,
    prod_pos,
    plot_SBM_low_carbon_preferences_timeseries
)

def double_plot_low_carbon_preferences_timeseries(
    fileName, 
    data_double, 
    dpi_save,
    tau_vals
    ):

    y_title = r"Low carbon preference, $A_{t,i,m}$"

    fig, axes = plt.subplots(nrows=2,ncols=data_double[0].M, sharey=True,constrained_layout=True,figsize=(10,6) )
    
    data_list_double = []
    for i,data in enumerate(data_double):
        data_list = []
        for v in range(data.N):
            data_indivdiual = np.asarray(data.agent_list[v].history_low_carbon_preferences)
            data_list.append(data_indivdiual)
            if data.M == 1:
                #print("data_indivdiual",data_indivdiual)
                #quit()
                axes[i].plot(
                        np.asarray(data.history_time),
                        data_indivdiual
                    )
            else:
                for j in range(data.M):
                    #print("HI", len(data.history_time), len(data_indivdiual[:,j]))
                    #quit()
                    axes[i][j].plot(
                        np.asarray(data.history_time),
                        data_indivdiual[:,j]
                    )
        data_list_double.append(data_list)

    data_list_array_double =np.asarray(data_list_double)#n,t,m
    #print("shape", data_list_array.shape)

    data_list_arr_t_double = np.transpose(data_list_array_double,(0,3,2,1))#tau,m,t,n

    #print("after", data_list_arr_t.shape)

    mean_data_double = np.mean(data_list_arr_t_double, axis = 3)
    median_data_double = np.median(data_list_arr_t_double, axis = 3)

    #print(" mean_data ", mean_data )
    for i, mean_data in enumerate(mean_data_double):
        axes[i].set_ylabel("Carbon price, $\\tau = %s$" % (tau_vals[i]))
        if data_double[0].M == 1:
            axes[i].plot(
                    np.asarray(data.history_time),
                    mean_data[0],
                    label= "mean",
                    linestyle="dotted"
                )
            axes[i].plot(
                    np.asarray(data.history_time),
                    median_data_double[i][0],
                    label= "median",
                    linestyle="dashed"
                )
            axes[i].legend()
        else:
            for j in range(data.M):
                axes[i][j].plot(
                    np.asarray(data.history_time),
                    mean_data[j],
                    label= "mean",
                    linestyle="dotted"
                )
                axes[i][j].plot(
                        np.asarray(data.history_time),
                        median_data_double[i][j],
                        label= "median",
                        linestyle="dashed"
                    )
                axes[i][j].legend()
                    
    fig.tight_layout()

    fig.supxlabel(r"Time, t")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/double_timeseries_preference"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    dpi_save = 600,
    ) -> None: 
    block_markers_list = ["o","s","^", "v","*","H","P","<",">"]#generate_scatter_markers(data.SBM_block_num)
    legend_loc = "upper right"
    lines_alpha = 0.2
    cmap_multi = get_cmap("plasma")
    cmap_weighting = get_cmap("Reds")

    norm_zero_one = Normalize(vmin=0, vmax=1)
    cmap = LinearSegmentedColormap.from_list(
        "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
    )
    node_sizes = 100   

    Data = load_object(fileName + "/Data", "social_networks")
    tau_vals = load_object(fileName + "/Data", "tau_vals")

    double_plot_low_carbon_preferences_timeseries(fileName, Data, dpi_save, tau_vals)
    #plot_network_start_preferences(fileName, Data,cmap, dpi_save, node_sizes,norm_zero_one)
    #plot_network_end_preferences(fileName, Data,cmap, dpi_save, node_sizes,norm_zero_one)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/time_brown_heg_17_28_57__01_03_2024",
    )


