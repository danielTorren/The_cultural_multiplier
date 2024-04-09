# imports
from cProfile import label
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from package.resources.utility import load_object
from package.resources.plot import (
    plot_end_points_emissions,
    plot_end_points_emissions_scatter,
    plot_end_points_emissions_lines,
)
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable

def plot_end_points_emissions_multi(
    fileName: str, Data_arr, property_title, property_save, property_vals, labels
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6),constrained_layout = True )

    for i, Data_list in enumerate(Data_arr):
        mu_emissions =  Data_list.mean(axis=1)
        min_emissions =  Data_list.min(axis=1)
        max_emissions=  Data_list.max(axis=1)

        ax.plot(property_vals, mu_emissions, label = labels[i])
        ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)

    ax.legend()
    ax.set_xlabel(property_title)
    ax.set_ylabel(r"Cumulative carbon emissions, E")

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/multi_" + property_save + "_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_end_points_emissions_multi_lines(
    fileName: str, Data_arr, property_title, property_save, property_vals, labels, colors_scenarios
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6),constrained_layout = True )

    for i, Data_list in enumerate(Data_arr):
        mu_emissions =  Data_list.mean(axis=1)
        #min_emissions =  Data_list.min(axis=1)
        #max_emissions=  Data_list.max(axis=1)

        ax.plot(property_vals, mu_emissions, label = labels[i], color = colors_scenarios[i])
        #ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)

        Data_list_trans = Data_list.T
        for v in range(len(Data_list_trans)):#loop through seeds
            ax.plot(property_vals, Data_list_trans[v], alpha = 0.2, color = colors_scenarios[i])#BLUE emisisons both sectors 2 sector

    ax.legend()
    ax.set_xlabel(property_title)
    ax.set_ylabel(r"Cumulative carbon emissions, E")

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/multi_lines_" + property_save + "_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_end_points_emissions_multi_blocks_lines(
    fileName: str, Data_arr,emissions_array_blocks, property_title, property_save, property_vals, labels, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows = 1, ncols = 3,figsize=(10,6),constrained_layout = True, sharey = True )
    
    data_blocks = np.transpose(emissions_array_blocks, (3, 0,1,2))
    data_block_1 = data_blocks[0]
    data_block_2 = data_blocks[1]

    for i, Data_list in enumerate(Data_arr):
        mu_emissions =  Data_list.mean(axis=1)
        data_run_1 = data_block_1[i].mean(axis=1)
        data_run_2 = data_block_2[i].mean(axis=1)
        #min_emissions =  Data_list.min(axis=1)
        #max_emissions=  Data_list.max(axis=1)

        axes[0].plot(property_vals, mu_emissions, label = labels[i], color = colors_scenarios[i])
        axes[1].plot(property_vals, data_run_1, label = labels[i], color = colors_scenarios[i])
        axes[2].plot(property_vals, data_run_2, label = labels[i], color = colors_scenarios[i])
        #ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)

        Data_list_trans = Data_list.T
        data1_trans = data_block_1[i].T
        data2_trans = data_block_2[i].T
        for v in range(len(Data_list_trans)):#loop through seeds
            axes[0].plot(property_vals, Data_list_trans[v], alpha = 0.2, color = colors_scenarios[i])
            axes[1].plot(property_vals, data1_trans[v], alpha = 0.2, color = colors_scenarios[i])
            axes[2].plot(property_vals, data2_trans[v], alpha = 0.2, color = colors_scenarios[i])
    axes[2].legend()
    axes[0].set_xlabel(property_title)
    axes[1].set_xlabel(property_title)
    axes[2].set_xlabel(property_title)

    axes[0].set_title("Total network")
    axes[1].set_title("Block 1")
    axes[2].set_title("Block 2")

    axes[0].set_ylabel(r"Cumulative carbon emissions, E")

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/multi_blocks_lines_" + property_save + "_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    dpi_save = 600,
    ) -> None: 

    ############################
    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    base_params = load_object(fileName + "/Data", "base_params")
    print("base_params",base_params)
    var_params  = load_object(fileName + "/Data" , "var_params")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    
    labels = [
        r"$\sigma_1 = 1.1,\sigma_2 = 1.1$, $h = 0$",
        r"$\sigma_1 = 1.1,\sigma_2 = 1.1$, $h = 1$(Block 1 Green, Block 2 Brown)",
        r"$\sigma_1 = 8,\sigma_2 = 8$, $h = 0$",
        r"$\sigma_1 = 8,\sigma_2 = 8$, $h = 1$(Block 1 Green, Block 2 Brown)",
        r"$\sigma_1 = 8,\sigma_2 = 1.5$, $h = 0$",
        r"$\sigma_1 = 8,\sigma_2 = 1.5$, $h = 1$(Block 1 Green, Block 2 Brown)",
        r"$\sigma_1 = 1.5,\sigma_2 = 8$, $h = 1$(Block 1 Green, Block 2 Brown)"
        ]

    property_varied = var_params["property_varied"]

    emissions_array = load_object(fileName + "/Data", "emissions_array")
    emissions_array_blocks = load_object(fileName + "/Data", "emissions_array_blocks")
    

    #PLOT THE 7 CASES one one plot
    #PLOT THE EMISSIOSN FOR DIFFERNT BLOCKS ON DIFFERENT SUBPLOTS
    #plot_end_points_emissions_multi(fileName, emissions_array, r"Carbon price, $\tau$", property_varied, property_values_list, labels)
    #plot_end_points_emissions_multi_lines(fileName, emissions_array, r"Carbon price, $\tau$", property_varied, property_values_list, labels,colors_scenarios)
    plot_end_points_emissions_multi_blocks_lines(fileName, emissions_array, emissions_array_blocks, r"Carbon price, $\tau$", property_varied, property_values_list, labels,colors_scenarios)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/SBM_BLOCK_sub_tau_vary_10_12_27__09_04_2024",
    )

