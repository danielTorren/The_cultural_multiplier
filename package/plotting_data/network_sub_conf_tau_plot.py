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
from matplotlib.colors import LinearSegmentedColormap, Normalize

def plot_means_end_points_emissions(
    fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, axis_val
):

    ncols = 3 
    nrows = 2
    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(15,6), constrained_layout = True)# #, sharey="col"

    #print( emissions_networks.shape)
    #quit()

    
    #shapre is 2,3,row,cols, seed
    cmap = get_cmap("plasma")
    if axis_val:
        c = Normalize()(property_values_list_row)
        ticks = np.linspace(min(property_values_list_col), max(property_values_list_col), 10)
        ticks_round = [round(i,2) for i in ticks]
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

                    axes[i][j].plot(property_values_list_col, mu_emissions,color=cmap(c[k]))#, c = colors[k]
                    axes[i][j].fill_between(property_values_list_col, min_emissions, max_emissions, alpha=0.4, facecolor=cmap(c[k]))#, facecolor = colors[k]
                    axes[i][j].set_xticks(ticks_round)
                    axes[i][j].set_xlim(min(property_values_list_col), max(property_values_list_col))
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( property_values_list_row), vmax=max(property_values_list_row))), ax=axes.ravel()
        )
        cbar.set_label(r"Confirmation bias, $\theta$")
        fig.supxlabel(r"Elasticity of substitution, $\sigma_m$")
    else:
        emissions_networks = np.transpose(emissions_networks, (0,1,3,2,4))
        c = Normalize()(property_values_list_col)
        for i in range(nrows):
            axes[i][0].set_ylabel(scenarios_titles[i])
            for j in range(ncols):
                axes[0][j].set_title(network_titles[j])
                for k in range(len(property_values_list_col)):
                    axes[i][j].grid()
                    #color = next(colors)#set color for whole scenario?
                    Data = emissions_networks[i][j][k]
                    #print("Data", Data.shape)
                    mu_emissions =  Data.mean(axis=1)
                    min_emissions =  Data.min(axis=1)
                    max_emissions=  Data.max(axis=1)

                    axes[i][j].plot(property_values_list_row, mu_emissions,color=cmap(c[k]))#, c = colors[k]
                    axes[i][j].fill_between(property_values_list_row, min_emissions, max_emissions, alpha=0.4, facecolor=cmap(c[k]))#, facecolor = colors[k]
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( property_values_list_col), vmax=max(property_values_list_col))), ax=axes.ravel()
        )
        cbar.set_label(r"Elasticity of substitution, $\sigma_m$")
        fig.supxlabel(r"Confirmation bias, $\theta$")

    fig.supylabel(r"Cumulative carbon emissions, E")

    #axes[0][2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_means_end_points_emissions_%s" %(axis_val)
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_means_no_bands_end_points_emissions(
    fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, axis_val
):

    ncols = 3 
    nrows = 2
    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(15,6), constrained_layout = True)# #, sharey="col"

    #print( emissions_networks.shape)
    #quit()

    #shapre is 2,3,row,cols, seed
    cmap = get_cmap("plasma")
    if axis_val:
        c = Normalize()(property_values_list_row)
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
                    #min_emissions =  Data.min(axis=1)
                    #max_emissions=  Data.max(axis=1)

                    axes[i][j].plot(property_values_list_col, mu_emissions,color=cmap(c[k]))#, c = colors[k]
                    #axes[i][j].fill_between(property_values_list_col, min_emissions, max_emissions, alpha=0.4, facecolor=cmap(c[k]))#, facecolor = colors[k]
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( property_values_list_row), vmax=max(property_values_list_row))), ax=axes.ravel()
        )
        cbar.set_label(r"Confirmation bias, $\theta$")
        fig.supxlabel(r"Elasticity of substitution, $\sigma_m$")
    else:
        emissions_networks = np.transpose(emissions_networks, (0,1,3,2,4))
        c = Normalize()(property_values_list_col)
        for i in range(nrows):
            axes[i][0].set_ylabel(scenarios_titles[i])
            for j in range(ncols):

                axes[0][j].set_title(network_titles[j])
                for k in range(len(property_values_list_col)):
                    axes[i][j].grid()
                    #color = next(colors)#set color for whole scenario?
                    Data = emissions_networks[i][j][k]
                    #print("Data", Data.shape)
                    mu_emissions =  Data.mean(axis=1)
                    #min_emissions =  Data.min(axis=1)
                    #max_emissions=  Data.max(axis=1)

                    axes[i][j].plot(property_values_list_row, mu_emissions,color=cmap(c[k]))#, c = colors[k]
                    #axes[i][j].fill_between(property_values_list_row, min_emissions, max_emissions, alpha=0.4, facecolor=cmap(c[k]))#, facecolor = colors[k]
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( property_values_list_col), vmax=max(property_values_list_col))), ax=axes.ravel()
        )
        cbar.set_label(r"Elasticity of substitution, $\sigma_m$")
        fig.supxlabel(r"Confirmation bias, $\theta$")

    fig.supylabel(r"Cumulative carbon emissions, E")

    #axes[0][2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_means_no_bands_end_points_emissions_%s" %(axis_val)
    fig.savefig(f+ ".png", dpi=600, format="png") 


def plot_price_elasticies_mean(fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, row_titles, colors):

    ncols = 3 
    nrows = 2
    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(15,6), constrained_layout = True)#, sharey=True

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
                
                axes[i][j].plot(property_values_list_col[1:], mu_emissions, label=row_titles[k])#, c = colors[k]
                axes[i][j].fill_between(property_values_list_col[1:], min_emissions, max_emissions, alpha=0.4)# facecolor = colors[k]
        
    fig.supxlabel(r"Elasticity of substitution, $\sigma_m$")
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    #axes[0][2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/plot_price_elasticies_mean"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_contour_emissions(
    fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, levels
):

    ncols = len(emissions_networks[0])
    nrows = len(emissions_networks)
    fig, axes = plt.subplots(nrows = nrows ,ncols =ncols,figsize=(15, 7), constrained_layout=True)

    for i, emissions in enumerate(emissions_networks):
        axes[i][0].set_ylabel(scenarios_titles[i])
        #axes[i][0].set_ylabel(r"Elasticity of substitution, $\sigma_m$")
        for j, emissions_network_specific in enumerate(emissions):
            axes[0][j].set_title(network_titles[j])
            Z = np.mean(emissions_network_specific, axis = 2)

            cmap = get_cmap("viridis_r")
            X, Y = np.meshgrid(property_values_list_col, property_values_list_row)
            #axes[i][j].set_xscale('log')
            cp = axes[i][j].contourf(X, Y, Z, cmap=cmap, levels = levels)
            cbar = fig.colorbar(
                cp,
                ax=axes[i][j],
            )
            if j == ncols-1:
                cbar.set_label(r"Cumulative carbon emissions, E")

    
    fig.supxlabel(r"Elasticity of substitution, $\sigma_m$")
    fig.supylabel(r"Confirmation bias, $\theta$")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_contour_emissions"
    #fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_scatter_emissions(
    fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, marker_size
):

    ncols = len(emissions_networks[0])
    nrows = len(emissions_networks)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 7), constrained_layout=True)

    for i, emissions in enumerate(emissions_networks):
        axes[i][0].set_ylabel(scenarios_titles[i])
        for j, emissions_network_specific in enumerate(emissions):
            axes[0][j].set_title(network_titles[j])
            Z = np.mean(emissions_network_specific, axis=2)

            X, Y = np.meshgrid(property_values_list_col, property_values_list_row)

            # Flatten X, Y, and Z for scatter plot
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = Z.flatten()
            #axes[i][j].set_xscale('log')
            # Scatter plot
            scatter = axes[i][j].scatter(X_flat, Y_flat, c=Z_flat, cmap="viridis_r", s=marker_size)

            # Colorbar
            cbar = fig.colorbar(scatter, ax=axes[i][j])
            if j == ncols - 1:
                cbar.set_label(r"Cumulative carbon emissions, E")

    fig.supxlabel(r"Elasticity of substitution, $\sigma_m$")
    fig.supylabel(r"Confirmation bias, $\theta$")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_scatter_emissions"
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_contour_price_elasticity(
    fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, levels
):

    ncols = len(emissions_networks[0])
    nrows = len(emissions_networks)
    fig, axes = plt.subplots(nrows = nrows ,ncols =ncols,figsize=(15, 7), constrained_layout=True)

    for i, emissions in enumerate(emissions_networks):
        axes[i][0].set_ylabel(scenarios_titles[i])
        #axes[i][0].set_ylabel(r"Elasticity of substitution, $\sigma_m$")
        for j, emissions_network_specific in enumerate(emissions):
            axes[0][j].set_title(network_titles[j])
            Z = np.mean(emissions_network_specific, axis = 2)
                
            data_trans_full = []
            for q in range(len(Z)):
                data_trans_full.append(calculate_price_elasticity(property_values_list_col, Z[q]))
            Data = np.asarray(data_trans_full).T
            #print(Data.shape)
            #quit()

            cmap = get_cmap("viridis_r")
            X, Y = np.meshgrid(property_values_list_col, property_values_list_row[1:])
            #axes[i][j].set_xscale('log')
            cp = axes[i][j].contourf(X, Y, Data, cmap=cmap, levels = levels)
            cbar = fig.colorbar(
                cp,
                ax=axes[i][j],
            )
            if j == ncols-1:
                cbar.set_label(r"Price elasticity of emissions, $\epsilon$")

    
    fig.supxlabel(r"Elasticity of substitution, $\sigma_m$")
    fig.supylabel(r"Confirmation bias, $\theta$")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_contour_price"
    #fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_scatter_price_elasticity(
    fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, marker_size
):

    ncols = len(emissions_networks[0])
    nrows = len(emissions_networks)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 7), constrained_layout=True)

    for i, emissions in enumerate(emissions_networks):
        axes[i][0].set_ylabel(scenarios_titles[i])
        for j, emissions_network_specific in enumerate(emissions):
            axes[0][j].set_title(network_titles[j])
            Z = np.mean(emissions_network_specific, axis=2)

            X, Y = np.meshgrid(property_values_list_col, property_values_list_row)

            # Flatten X, Y, and Z for scatter plot
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = Z.flatten()

            #axes[i][j].set_xscale('log')
            # Scatter plot
            scatter = axes[i][j].scatter(X_flat, Y_flat, c=Z_flat, cmap="viridis_r", s=marker_size)

            # Colorbar
            cbar = fig.colorbar(scatter, ax=axes[i][j])
            if j == ncols - 1:
                cbar.set_label(r"Cumulative carbon emissions, E")

    fig.supxlabel(r"Elasticity of substitution, $\sigma_m$")
    fig.supylabel(r"Confirmation bias, $\theta$")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_scatter_price"
    fig.savefig(f + ".png", dpi=600, format="png")

# Example usage:
# plot_scatter(fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, levels)


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
    scenario_labels = [r"Low carbon price, $\tau = 0.1$", r"High carbon price, $\tau = 1.0$"]
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    print("variable_parameters_dict",variable_parameters_dict)
    base_params = load_object(fileName + "/Data", "base_params") 
    print("base_params", base_params)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]
    row_label = row_dict["title"]#r"Attitude Beta parameters, $(a,b)$"#r"Number of behaviours per agent, M"
    col_label = col_dict["title"]#r'Confirmation bias, $\theta$'
    property_values_list_col = col_dict["vals"]
    
    property_values_list_row = row_dict["vals"]

    #row_titles = [r"Confirmation bias, $\theta$ = %s" % (i) for i in property_values_list_row]
    #print("row_titles",row_titles)
    #EMISSIONS PLOTS ALL TOGETHER SEEDS
    #plot_scatter_end_points_emissions_scatter(fileName, emissions_networks, scenario_labels ,property_values_list_col, property_values_list_row,network_titles,colors_scenarios)
                                    #fileName, emissions_networks, scenarios_titles, property_values_list_col, property_values_list_row, network_titles, row_titles, colors
    plot_means_end_points_emissions(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,0)
    plot_means_end_points_emissions(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,1)
    plot_means_no_bands_end_points_emissions(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,0)
    plot_means_no_bands_end_points_emissions(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,1)

    plot_contour_emissions(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,50)
    plot_scatter_emissions(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,2)

    #plot_contour_price_elasticity(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,20)
    #plot_scatter_price_elasticity(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,2)

    #PRICE ELASTICITIES
    #plot_price_elasticies_seeds(fileName, emissions_networks, scenario_labels,property_values_list_col, property_values_list_row,seed_reps,seeds_to_show,network_titles,colors_scenarios)
    #DO THIS
    #plot_price_elasticies_mean(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,row_titles,colors)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/network_sub_conf_tau_13_09_11__20_02_2024"
    )