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

def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    dpi_save = 600,
    ) -> None: 

    ############################

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
    plot_end_points_emissions_multi(fileName, emissions_array, r"Carbon price, $\tau$", property_varied, property_values_list, labels)

    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/SBM_BLOCK_sub_tau_vary_19_15_36__31_01_2024",
    )
