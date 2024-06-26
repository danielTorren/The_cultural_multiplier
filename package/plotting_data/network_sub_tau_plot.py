"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports

import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object,
)
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

def plot_means_end_points_emissions(
    fileName, emissions_networks, property_values_list_col, property_values_list_row, network_titles, row_titles, colors
):

    ncols = 3 
    nrows = 1
    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(15,6), constrained_layout = True)# 

    for j in range(ncols):
        axes[j].set_title(network_titles[j])
        for k in range(len(property_values_list_row)):
            axes[j].grid()
            Data = emissions_networks[j][k]
            #print("Data", Data.shape)
            mu_emissions =  Data.mean(axis=1)
            axes[j].plot(property_values_list_col, mu_emissions, label=row_titles[k], c = colors[k])

            Data_trans = Data.T
            for v, data_run in enumerate(Data_trans):
                axes[j].plot(property_values_list_col, data_run, c = colors[k], alpha = 0.1)

        
    fig.supxlabel(r"Carbon price, $\tau$")
    fig.supylabel(r"Cumulative carbon emissions, E")

    axes[2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_sub_tau_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def main(
    fileName = "results/tax_sweep_11_29_20__28_09_2023"
) -> None:
    
    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors= cmap.colors  # type: list
    #print(colors)

    #quit()
    emissions_networks = load_object(fileName + "/Data","emissions_data_3")
    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    base_params = load_object(fileName + "/Data", "base_params") 
    print("base_params",base_params)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]
    row_label = row_dict["title"]#r"Attitude Beta parameters, $(a,b)$"#r"Number of behaviours per agent, M"
    col_label = col_dict["title"]#r'Confirmation bias, $\theta$'
    property_values_list_col = col_dict["vals"]
    property_values_list_row = row_dict["vals"]

    row_titles = ["Elasticity of substitution, $\sigma_m$ = %s" % (round(i,3)) for i in property_values_list_row]
    plot_means_end_points_emissions(fileName, emissions_networks, property_values_list_col, property_values_list_row,network_titles,row_titles,colors)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/network_sub_tau_15_32_07__09_04_2024"
    )