"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports

import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object,
)
from package.plotting_data.price_elasticity_plot import calculate_price_elasticity,calc_price_elasticities_2D
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

def plot_means_end_points_emissions(
    fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, row_titles, colors
):

    ncols = 3 
    nrows = 2
    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(15,6), constrained_layout = True, sharey="col")# 

    #shapre is 2,3,row,cols, seed
    for i in range(nrows):
        axes[i][0].set_ylabel(scenarios_titles[i])
        for j in range(ncols):
            axes[0][j].set_title(network_titles[j])
            for k in range(len(property_values_list_row)):
                axes[i][j].grid()
                #color = next(colors)#set color for whole scenario?
                Data = emissions_networks[i][j][k]
                #print("Data", Data.shape)
                mu_emissions =  Data.mean(axis=1)
                min_emissions =  Data.min(axis=1)
                max_emissions=  Data.max(axis=1)

                axes[i][j].plot(property_values_list_col, mu_emissions, label=row_titles[k], c = colors[k])
                axes[i][j].fill_between(property_values_list_col, min_emissions, max_emissions, alpha=0.4, facecolor = colors[k])
        
    fig.supxlabel(r"Carbon price, $\tau$")
    fig.supylabel(r"Cumulative carbon emissions, E")

    axes[0][2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_means_end_points_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png") 


def plot_price_elasticies_mean(fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, row_titles, colors):

    ncols = 3 
    nrows = 2
    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(15,6), constrained_layout = True, sharey=True)

    #shapre is 2,3,row,cols, seed
    for i in range(nrows):
        axes[i][0].set_ylabel(scenarios_titles[i])
        for j in range(ncols):
            axes[0][j].set_title(network_titles[j])
            for k in range(len(property_values_list_row)):
                #color = next(colors)#set color for whole scenario?
                axes[i][j].grid()
                Data = emissions_networks[i][j][k]
                data_trans_full = []
                for q in range(len(Data)):
                    data_trans_full.append(calculate_price_elasticity(property_values_list_col, Data[q]))
                Data = np.asarray(data_trans_full).T
                #print("Data", Data.shape)
                mu_emissions =  Data.mean(axis=1)
                min_emissions =  Data.min(axis=1)
                max_emissions=  Data.max(axis=1)
                
                axes[i][j].plot(property_values_list_col[1:], mu_emissions, label=row_titles[k], c = colors[k])
                axes[i][j].fill_between(property_values_list_col[1:], min_emissions, max_emissions, alpha=0.4, facecolor = colors[k])
        
    fig.supxlabel(r"Carbon price, $\tau$")
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    axes[0][2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/plot_price_elasticies_mean"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def main(
    fileName = "results/tax_sweep_11_29_20__28_09_2023"
) -> None:
    
    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors= cmap.colors  # type: list
    #print(colors)

    #quit()
    emissions_networks = load_object(fileName + "/Data","emissions_data_2_3")
    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    scenario_labels = ["Dynamic social weighting", "Dynamic identity weighting"]
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
    print("row_titles",row_titles)
    #EMISSIONS PLOTS ALL TOGETHER SEEDS
    #plot_scatter_end_points_emissions_scatter(fileName, emissions_networks, scenario_labels ,property_values_list_col, property_values_list_row,network_titles,colors_scenarios)
                                    #fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, row_titles, colors
    plot_means_end_points_emissions(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,row_titles,colors)

    #PRICE ELASTICITIES
    #plot_price_elasticies_seeds(fileName, emissions_networks, scenario_labels,property_values_list_col, property_values_list_row,seed_reps,seeds_to_show,network_titles,colors_scenarios)
    #plot_price_elasticies_mean(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,row_titles,colors)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/network_sub_tau_16_06_37__08_04_2024"
    )