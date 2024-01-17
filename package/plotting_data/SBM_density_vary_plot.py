"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

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

def plot_relative_end_points_emissions_two(
    fileName: str, Data_arr_no_homophily, Data_arr_homophily, property_title, property_save, property_vals, labels
):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), constrained_layout=True, sharey=True)

    data = [Data_arr_no_homophily, Data_arr_homophily]
    ax_titles = [r"No identity homophily, h = 0", r"High identity homophily, h = 1"]

    for j, ax in enumerate(axes.flat):
        Data_arr = data[j]

        for i, Data_list in enumerate(Data_arr):
            mu_emissions = Data_list.mean(axis=1)
            min_emissions = Data_list.min(axis=1)
            max_emissions = Data_list.max(axis=1)

            # Calculate percentage increase relative to the first value
            mu_relative_emissions = (mu_emissions / mu_emissions[0])
            min_relative_emissions = (min_emissions / mu_emissions[0])
            max_relative_emissions = (max_emissions / mu_emissions[0])

            ax.plot(property_vals, mu_relative_emissions, label=labels[i])
            ax.fill_between(
                property_vals,
                min_relative_emissions,
                max_relative_emissions,
                alpha=0.5,
            )

        ax.legend()
        ax.set_xlabel(property_title)
        ax.set_title(ax_titles[j])

    axes[0].set_ylabel(r"Relative Cumulative Carbon Emissions ratio $E/E_{\tau =0}$")

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/multi_" + property_save + "_relative_emissions"
    fig.savefig(f + ".png", dpi=600, format="png")


def plot_end_points_emissions_two(
    fileName: str, Data_arr_no_homophily,Data_arr_homophily, property_title, property_save, property_vals, labels
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10,6),constrained_layout = True, sharey=True )

    data = [Data_arr_no_homophily,Data_arr_homophily]
    ax_titles = [r"No identity homophily, h = 0", r"High identity homophily, h = 1"]
    for j,ax in enumerate(axes.flat):

        Data_arr = data[j]
        for i, Data_list in enumerate(Data_arr):
            mu_emissions =  Data_list.mean(axis=1)
            min_emissions =  Data_list.min(axis=1)
            max_emissions=  Data_list.max(axis=1)

            ax.plot(property_vals, mu_emissions, label = labels[i])
            ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)

        ax.legend()
        ax.set_xlabel(property_title)
        
        ax.set_title(ax_titles[j])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
        
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/multi_" + property_save + "_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    dpi_save = 600,
    ) -> None: 

    ############################

    base_params = load_object(fileName + "/Data", "base_params")
    print("base_params",base_params)
    var_params  = load_object(fileName + "/Data" , "var_params")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    labels = [r"No carbon price, $\tau = 0$", r"Low carbon price, $\tau = 0.1$", r"High carbon price, $\tau = 0.5$"]

    property_varied = var_params["property_varied"]

    emissions_array_no_homophilly = load_object(fileName + "/Data", "emissions_array_no_homophilly")
    emissions_array_homophilly = load_object(fileName + "/Data", "emissions_array_homophilly")
        
    plot_end_points_emissions_two(fileName,emissions_array_no_homophilly, emissions_array_homophilly, r"Inter-block link density", property_varied, property_values_list, labels)
    plot_relative_end_points_emissions_two(fileName,emissions_array_no_homophilly, emissions_array_homophilly, r"Inter-block link density", property_varied, property_values_list, labels)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/SBM_density_vary_13_38_00__16_01_2024",
    )

