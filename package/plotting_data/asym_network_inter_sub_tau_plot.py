"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports

import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object,
    calc_bounds
)
from matplotlib.colors import LinearSegmentedColormap, Normalize
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

    #print(" scenarios_titles", scenarios_titles)
    #print(" network_titles, row_titles", network_titles, row_titles)
    #shapre is 2,3,row,cols, seed
    for i in range(nrows):
        axes[i][0].set_ylabel(scenarios_titles[i])
        for j in range(ncols):
            axes[0][j].set_title(network_titles[j])
            for k in range(len(property_values_list_row)):
                #print("k",k)
                axes[i][j].grid()
                #color = next(colors)#set color for whole scenario?
                Data = emissions_networks[i][j][k]
                #print("Data", Data.shape)
                mu_emissions =  Data.mean(axis=1)
                min_emissions =  Data.min(axis=1)
                max_emissions=  Data.max(axis=1)
                #print(len(row_titles), len(colors))
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

def multi_line_matrix_plot_stoch_bands_23(
    fileName, emissions_networks, col_vals, row_vals,  Y_param, cmap, dpi_save, col_axis_x, col_label, row_label, y_label, scenarios_titles,network_titles
    ):
    
    ncols = 3 
    nrows = 2

    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, constrained_layout=True,figsize=(20, 7))#
    #cmap = plt.get_cmap("cividis")

    for i in range(nrows):
        axes[i][0].set_ylabel(scenarios_titles[i])
        for j in range(ncols):
            axes[0][j].set_title(network_titles[j])
            Z_array = emissions_networks[i][j]
            if col_axis_x:#= 1
                c = Normalize()(row_vals)
                for k in range(len(Z_array)):
                    data = Z_array[k]#(sigma, seeds)
                    ys_mean = data.mean(axis=1)
                    ys_min = data.min(axis=1)
                    ys_max= data.max(axis=1)

                    axes[i][j].plot(col_vals, ys_mean, ls="-", linewidth = 0.5, color = cmap(c[k]))
                    axes[i][j].fill_between(col_vals, ys_min, ys_max, facecolor=cmap(c[k]), alpha=0.5)
                cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( row_vals), vmax=max(row_vals))), ax=axes[i][j]
            )
            else:
                Z_array_T = np.transpose(Z_array,(1,0,2))#put (sigma, tau,seeds) from (tau,sigma, seeds)
                c = Normalize()(col_vals)
                for k in range(len(Z_array_T)):#loop through sigma
                    data = Z_array_T[k]
                    ys_mean = data.mean(axis=1)
                    ys_min = data.min(axis=1)
                    ys_max= data.max(axis=1)

                    axes[i][j].plot(row_vals, ys_mean, ls="-", linewidth = 0.5,color=cmap(c[k]))
                    axes[i][j].fill_between(row_vals, ys_min, ys_max, facecolor=cmap(c[k]), alpha=0.5)

                cbar = fig.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( col_vals), vmax=max(col_vals))), ax=axes[i][j]
                )   

            #quit()
                            
            fig.supylabel(r"Cumulative carbon emissions, E")
            #axes[i][j].set_ylabel(y_label)#(r"First behaviour attitude variance, $\sigma^2$")

            if col_axis_x:
                fig.supxlabel(col_label)#(r"Carbon price, $\tau$")
                cbar.set_label(row_label)#(r"Number of behaviours per agent, M")
                #axes[i][j].set_xlabel(col_label)#(r'Confirmation bias, $\theta$')
            else:
                cbar.set_label(col_label)#)(r'Confirmation bias, $\theta$')
                #axes[i][j].set_xlabel(row_label)#(r"Number of behaviours per agent, M")
                fig.supxlabel(row_label)
    plotName = fileName + "/Plots"
    f = plotName + "/23_multi_line_matrix_plot_stoch_fill_%s_%s" % (Y_param, col_axis_x)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")  


def multi_line_matrix_plot_stoch_bands_23_95_confidence(
    fileName, emissions_networks, col_vals, row_vals,  Y_param, cmap, dpi_save, col_axis_x, col_label, row_label, y_label, scenarios_titles,network_titles, confidence_level
    ):
    
    ncols = 3 
    nrows = 2

    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, constrained_layout=True,figsize=(20, 7))#
    #cmap = plt.get_cmap("cividis")

    for i in range(nrows):
        axes[i][0].set_ylabel(scenarios_titles[i])
        for j in range(ncols):
            axes[0][j].set_title(network_titles[j])
            Z_array = emissions_networks[i][j]
            if col_axis_x:#= 1
                c = Normalize()(row_vals)
                for k in range(len(Z_array)):
                    data = Z_array[k]#(sigma, seeds)
                    ys_mean, ys_min, ys_max = calc_bounds(data, confidence_level)
                    #ys_mean = data.mean(axis=1)
                    #ys_min = data.min(axis=1)
                    #test_ys_max= data.max(axis=1)
                    #print("difference", test_ys_max,ys_max,test_ys_max- ys_max )
                    #quit()

                    axes[i][j].plot(col_vals, ys_mean, ls="-", linewidth = 0.5, color = cmap(c[k]))
                    axes[i][j].fill_between(col_vals, ys_min, ys_max, facecolor=cmap(c[k]), alpha=0.5)
                cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( row_vals), vmax=max(row_vals))), ax=axes[i][j]
            )
            else:
                Z_array_T = np.transpose(Z_array,(1,0,2))#put (sigma, tau,seeds) from (tau,sigma, seeds)
                c = Normalize()(col_vals)
                for k in range(len(Z_array_T)):#loop through sigma
                    data = Z_array_T[k]

                    ys_mean, ys_min, ys_max = calc_bounds(data, confidence_level)
                    #ys_mean = data.mean(axis=1)
                    #ys_min = data.min(axis=1)
                    test_ys_max= data.max(axis=1)
                    print("difference", test_ys_max,ys_max,test_ys_max- ys_max )
                    quit()
                    axes[i][j].plot(row_vals, ys_mean, ls="-", linewidth = 0.5,color=cmap(c[k]))
                    axes[i][j].fill_between(row_vals, ys_min, ys_max, facecolor=cmap(c[k]), alpha=0.5)

                cbar = fig.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( col_vals), vmax=max(col_vals))), ax=axes[i][j]
                )   

            #quit()
                            
            fig.supylabel(r"Cumulative carbon emissions, E")
            #axes[i][j].set_ylabel(y_label)#(r"First behaviour attitude variance, $\sigma^2$")

            if col_axis_x:
                fig.supxlabel(col_label)#(r"Carbon price, $\tau$")
                cbar.set_label(row_label)#(r"Number of behaviours per agent, M")
                #axes[i][j].set_xlabel(col_label)#(r'Confirmation bias, $\theta$')
            else:
                cbar.set_label(col_label)#)(r'Confirmation bias, $\theta$')
                #axes[i][j].set_xlabel(row_label)#(r"Number of behaviours per agent, M")
                fig.supxlabel(row_label)
    plotName = fileName + "/Plots"
    f = plotName + "/confidence_23_multi_line_matrix_plot_stoch_fill_%s_%s" % (Y_param, col_axis_x)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")  


def multi_line_matrix_plot_stoch_bands_23_95_confidence_inset(
    fileName, emissions_networks, col_vals, row_vals,  Y_param, cmap, dpi_save, col_axis_x, col_label, row_label, y_label, scenarios_titles,network_titles, confidence_level, min_val_inset
    ):
    
    ncols = 3 
    nrows = 2

    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, constrained_layout=True,figsize=(20, 7))#
   
    #cmap = plt.get_cmap("cividis")

    for i in range(nrows):
        axes[i][0].set_ylabel(scenarios_titles[i])
        for j in range(ncols):
            axes[0][j].set_title(network_titles[j])
             

            Z_array = emissions_networks[i][j]
            if col_axis_x:#= 1
                if i == 1:
                    inset_ax = axes[1][j].inset_axes([0.5, 0.5, 0.45, 0.45])

                c = Normalize()(row_vals)
                for k in range(len(Z_array)):
                    data = Z_array[k]#(sigma, seeds)
                    ys_mean, ys_min, ys_max = calc_bounds(data, confidence_level)
                    #ys_mean = data.mean(axis=1)
                    #ys_min = data.min(axis=1)
                    #test_ys_max= data.max(axis=1)
                    #print("difference", test_ys_max,ys_max,test_ys_max- ys_max )
                    #quit()

                    axes[i][j].plot(col_vals, ys_mean, ls="-", linewidth = 0.5, color = cmap(c[k]))
                    axes[i][j].fill_between(col_vals, ys_min, ys_max, facecolor=cmap(c[k]), alpha=0.5)

                    #INSET AXIS
                    if i == 1:       
                        index_min = np.where(col_vals> min_val_inset)[0][0]#get index where its more than 0.8 carbon tax, this way independent of number of reps as longas more than 3
                        #print(index_min)
                        #quit()                 
                        inset_ax.plot(col_vals[index_min:-1], ys_mean[index_min:-1], ls="-", linewidth = 0.5, color = cmap(c[k]))
                        inset_ax.fill_between(col_vals[index_min:-1], ys_min[index_min:-1], ys_max[index_min:-1], facecolor=cmap(c[k]), alpha=0.5)


                cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( row_vals), vmax=max(row_vals))), ax=axes[i][j]
            )
            else:
                Z_array_T = np.transpose(Z_array,(1,0,2))#put (sigma, tau,seeds) from (tau,sigma, seeds)
                c = Normalize()(col_vals)
                for k in range(len(Z_array_T)):#loop through sigma
                    data = Z_array_T[k]

                    ys_mean, ys_min, ys_max = calc_bounds(data, confidence_level)
                    #ys_mean = data.mean(axis=1)
                    #ys_min = data.min(axis=1)
                    test_ys_max= data.max(axis=1)
                    #print("difference", test_ys_max,ys_max,test_ys_max- ys_max )
                    #quit()
                    axes[i][j].plot(row_vals, ys_mean, ls="-", linewidth = 0.5,color=cmap(c[k]))
                    axes[i][j].fill_between(row_vals, ys_min, ys_max, facecolor=cmap(c[k]), alpha=0.5)

                cbar = fig.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( col_vals), vmax=max(col_vals))), ax=axes[i][j]
                )   

            #quit()
                            
            fig.supylabel(r"Cumulative carbon emissions, E")
            #axes[i][j].set_ylabel(y_label)#(r"First behaviour attitude variance, $\sigma^2$")

            if col_axis_x:
                fig.supxlabel(col_label)#(r"Carbon price, $\tau$")
                if j == ncols-1:
                    cbar.set_label(row_label)#(r"Number of behaviours per agent, M")
                #axes[i][j].set_xlabel(col_label)#(r'Confirmation bias, $\theta$')
            else:
                cbar.set_label(col_label)#)(r'Confirmation bias, $\theta$')
                #axes[i][j].set_xlabel(row_label)#(r"Number of behaviours per agent, M")
                fig.supxlabel(row_label)
    plotName = fileName + "/Plots"
    f = plotName + "/inset_confidence_23_multi_line_matrix_plot_stoch_fill_%s_%s" % (Y_param, col_axis_x)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")  




def main(
    fileName = "results/tax_sweep_11_29_20__28_09_2023"
) -> None:
    dpi_save = 600
    name = "Paired"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list
    #print(colors)

    confidence_level = 0.95
    min_val_inset = 0.5

    #quit()
    emissions_networks = load_object(fileName + "/Data","emissions_data_2_3")
    #print(emissions_networks.shape)
    #quit()
    
    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    scenario_labels = ["Intra-sector substitutability, $\\sigma_m = 1.5$", "Intra-sector substitutability, $\\sigma_m = 5$"]
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    base_params = load_object(fileName + "/Data", "base_params") 
    #print("base_params",base_params)
    #quit()
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
    #plot_means_end_points_emissions(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,row_titles,colors)

    #PRICE ELASTICITIES
    #plot_price_elasticies_seeds(fileName, emissions_networks, scenario_labels,property_values_list_col, property_values_list_row,seed_reps,seeds_to_show,network_titles,colors_scenarios)
    #plot_price_elasticies_mean(fileName, emissions_networks, scenario_labels, property_values_list_col, property_values_list_row,network_titles,row_titles,colors)

    col_dict = variable_parameters_dict["col"]
    #print("col dict",col_dict)
    #col_dict["vals"] = col_dict["vals"][:-1]
    #print("col dict",col_dict)
    row_dict = variable_parameters_dict["row"]
    #print("row dict",row_dict)
    #quit()

    row_label = row_dict["title"]#r"Attitude Beta parameters, $(a,b)$"#r"Number of behaviours per agent, M"
    col_label = col_dict["title"]#r'Confirmation bias, $\theta$'
    y_label = "Cumulative emissions, $E$"#col_dict["title"]#r"Identity variance, $\sigma^2$"
        
    multi_line_matrix_plot_stoch_bands_23(fileName, emissions_networks, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"), dpi_save, 0, col_label, row_label, y_label, scenario_labels,network_titles)
    multi_line_matrix_plot_stoch_bands_23(fileName, emissions_networks, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"), dpi_save, 1, col_label, row_label, y_label, scenario_labels,network_titles)
    #multi_line_matrix_plot_stoch_bands_23_95_confidence(fileName, emissions_networks, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"), dpi_save, 1, col_label, row_label, y_label, scenario_labels,network_titles,confidence_level)
    #multi_line_matrix_plot_stoch_bands_23_95_confidence_inset(fileName, emissions_networks, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"), dpi_save, 1, "Carbon price Sector 2, $\\tau_2$","Inter-sector substitutability, $\\nu$", "Cumulative carbon emissions, $E$", scenario_labels,network_titles,confidence_level,min_val_inset)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/asym_network_inter_sub_tau_22_31_21__06_04_2024"
    )