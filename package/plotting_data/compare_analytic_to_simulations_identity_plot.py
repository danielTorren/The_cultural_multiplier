"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports

import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object
)
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt


def plot_sector_rebound_simulations_identity(
    fileName, total_emissions_1_identity, total_emissions_2_identity,  emissions_networks_2_identity,  total_emissions_1_socially, total_emissions_2_socially, emissions_networks_2_socially, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 2,figsize=(15,8), sharey="col")#, constrained_layout = True

    for k in range(3):

        max_val_inset = 0.1
        inset_ax_identity = axes[0][k].inset_axes([0.4, 0.4, 0.55, 0.55])
        inset_ax_socially = axes[1][k].inset_axes([0.4, 0.4, 0.55, 0.55])
        index_min = np.where(property_vals > max_val_inset)[0][0]#get index where its more than 0.8 carbon tax, this way independent of number of reps as longas more than 3       

        emissions_1_identity  = total_emissions_1_identity[k].T
        emissions_2_identity  = total_emissions_2_identity[k].T
        emissiosn_sector_2_identity = np.transpose(emissions_networks_2_identity[k],(2,1,0))

        emissions_1_socially  = total_emissions_1_socially[k].T
        emissions_2_socially  = total_emissions_2_socially[k].T
        emissiosn_sector_2_socially = np.transpose(emissions_networks_2_socially[k],(2,1,0))

        
        emissions_1_mean_identity = (emissions_1_identity).mean(axis=0)
        emissions_2_mean_identity = (emissions_2_identity).mean(axis=0)
        emissiosn_sector_2_1_mean_identity = (emissiosn_sector_2_identity[0]).mean(axis=0)
        emissiosn_sector_2_2_mean_identity = (emissiosn_sector_2_identity[1]).mean(axis=0)

        emissions_1_mean_socially = (emissions_1_socially).mean(axis=0)
        emissions_2_mean_socially = (emissions_2_socially).mean(axis=0)
        emissiosn_sector_2_1_mean_socially = (emissiosn_sector_2_socially[0]).mean(axis=0)
        emissiosn_sector_2_2_mean_socially = (emissiosn_sector_2_socially[1]).mean(axis=0)


        #PLOT MEANS
        axes[0][k].plot(property_vals, emissions_2_mean_identity, alpha = 1, color = "Blue", label = r'Emissions both sectors, $E_F$')#BLUE emisisons both sectors 2 sector
        axes[0][k].plot(property_vals, emissions_1_mean_identity, alpha = 1, color = "Red", label = r'Emissions no sector 2, $E^{*}_{F}$', ls="--")#RED Dashed emisisons both sectors 2 sector
        
        axes[0][k].plot(property_vals, emissiosn_sector_2_1_mean_identity, alpha = 1, color = "Orange", label = r'Emissions sector 1, $E_{F,1}$')#ORANGE emssions from sector 1 of the 2 sector run
        axes[0][k].plot(property_vals, emissiosn_sector_2_2_mean_identity, alpha = 1, color = "Green", label = r'Emissions sector 2, $E_{F,2}$')#Green emssions from sector 2 of the 2 sector run        

        inset_ax_identity.plot(property_vals[:index_min], emissions_2_mean_identity[:index_min], color = "Blue")
        inset_ax_identity.plot(property_vals[:index_min], emissions_1_mean_identity[:index_min], color = "Red", ls="--")
        inset_ax_identity.plot(property_vals[:index_min], emissiosn_sector_2_1_mean_identity[:index_min], color = "Orange")
        inset_ax_identity.plot(property_vals[:index_min], emissiosn_sector_2_2_mean_identity[:index_min],  color = "Green")

        for v in range(len(emissions_1_identity)):#loop through seeds
            axes[0][k].plot(property_vals, emissions_2_identity[v], alpha = 0.2, color = "Blue")#BLUE emisisons both sectors 2 sector
            axes[0][k].plot(property_vals, emissions_1_identity[v], alpha = 0.2, color = "Red", ls="--")#RED Dashed emisisons both sectors 2 sector
            
            axes[0][k].plot(property_vals, emissiosn_sector_2_identity[0][v], alpha = 0.2, color = "Orange")#ORANGE emssions from sector 1 of the 2 sector run
            axes[0][k].plot(property_vals, emissiosn_sector_2_identity[1][v], alpha = 0.2, color = "Green")#Green emssions from sector 2 of the 2 sector run
            
            inset_ax_identity.plot(property_vals[:index_min], emissions_2_identity[v][:index_min], color = "Blue", alpha = 0.2)
            inset_ax_identity.plot(property_vals[:index_min], emissions_1_identity[v][:index_min], color = "Red", ls="--", alpha = 0.2)
            inset_ax_identity.plot(property_vals[:index_min], emissiosn_sector_2_identity[0][v][:index_min], color = "Orange", alpha = 0.2)
            inset_ax_identity.plot(property_vals[:index_min], emissiosn_sector_2_identity[1][v][:index_min],  color = "Green", alpha = 0.2)

        #ax.legend()
        axes[0][k].set_title (network_titles[k])


        
        #PLOT MEANS
        #SOCIAL
        axes[1][k].plot(property_vals, emissions_2_mean_socially, alpha = 1, color = "Blue", label = r'Emissions both sectors, $E_F$')#BLUE emisisons both sectors 2 sector
        axes[1][k].plot(property_vals, emissions_1_mean_socially, alpha = 1, color = "Red", label = r'Emissions no sector 2, $E^{*}_{F}$', ls="--")#RED Dashed emisisons both sectors 2 sector
            
        axes[1][k].plot(property_vals, emissiosn_sector_2_1_mean_socially, alpha = 1, color = "Orange", label = r'Emissions sector 1, $E_{F,1}$')#ORANGE emssions from sector 1 of the 2 sector run
        axes[1][k].plot(property_vals, emissiosn_sector_2_2_mean_socially, alpha = 1, color = "Green", label = r'Emissions sector 2, $E_{F,2}$')#Green emssions from sector 2 of the 2 sector run        

        inset_ax_socially.plot(property_vals[:index_min], emissions_2_mean_socially[:index_min], color = "Blue")
        inset_ax_socially.plot(property_vals[:index_min], emissions_1_mean_socially[:index_min], color = "Red", ls="--")
        inset_ax_socially.plot(property_vals[:index_min], emissiosn_sector_2_1_mean_socially[:index_min], color = "Orange")
        inset_ax_socially.plot(property_vals[:index_min], emissiosn_sector_2_2_mean_socially[:index_min],  color = "Green")


        for v in range(len(emissions_1_socially)):#loop through seeds
            axes[1][k].plot(property_vals, emissions_2_socially[v], alpha = 0.2, color = "Blue")#BLUE emisisons both sectors 2 sector
            axes[1][k].plot(property_vals, emissions_1_socially[v], alpha = 0.2, color = "Red", ls="--")#RED Dashed emisisons both sectors 2 sector
            axes[1][k].plot(property_vals, emissiosn_sector_2_socially[0][v], alpha = 0.2, color = "Orange")#ORANGE emssions from sector 1 of the 2 sector run
            axes[1][k].plot(property_vals, emissiosn_sector_2_socially[1][v], alpha = 0.2, color = "Green")#Green emssions from sector 2 of the 2 sector run
            
            inset_ax_socially.plot(property_vals[:index_min], emissions_2_socially[v][:index_min], color = "Blue", alpha = 0.2)
            inset_ax_socially.plot(property_vals[:index_min], emissions_1_socially[v][:index_min], color = "Red", ls="--", alpha = 0.2)
            inset_ax_socially.plot(property_vals[:index_min], emissiosn_sector_2_socially[0][v][:index_min], color = "Orange", alpha = 0.2)
            inset_ax_socially.plot(property_vals[:index_min], emissiosn_sector_2_socially[1][v][:index_min],  color = "Green", alpha = 0.2)

        #ax.legend()
        axes[1][k].set_xlabel(r"Carbon price, $\tau$")

    axes[0][0].set_ylabel("Dynamic identity weighting")
    axes[1][0].set_ylabel("Dynamic social weighting")
    fig.subplots_adjust(left=0.08)#move the axes a little to the left 
    #fig.supxlabel(r"Carbon price, $\tau$")
    fig.supylabel(r"Cumulative carbon emissions, E")
    

    #axes[0][2].legend(fontsize="8")
    handles, labels = axes[0][2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize="10")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_2_sector_rebound_simulations_identity"
    fig.savefig(f+ ".png", dpi=600, format="png") 


def main(
    fileName ,
) -> None:
    
    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list
    #print(colors)

    #quit()
    data_array_1_identity = load_object(fileName + "/Data", "data_array_1_identity")
    data_array_2_identity = load_object(fileName + "/Data", "data_array_2_identity")
    data_sectors_2_identity = load_object(fileName + "/Data", "data_sectors_2_identity")

    data_array_1_socially = load_object(fileName + "/Data", "data_array_1_socially")
    data_array_2_socially = load_object(fileName + "/Data", "data_array_2_socially")
    data_sectors_2_socially = load_object(fileName + "/Data", "data_sectors_2_socially")
    
    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    scenario_labels = ["One sector, $M = 1$", "Two sectors, $M = 2$"]
    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 
    print("base params", base_params)

    #EMISSIONS PLOTS ALL TOGETHER SEEDS
    plot_sector_rebound_simulations_identity(fileName, data_array_1_identity, data_array_2_identity, data_sectors_2_identity,   data_array_1_socially, data_array_2_socially, data_sectors_2_socially, scenario_labels ,property_values_list,network_titles,colors_scenarios)

    #plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/identity_compare_analytic_emissions_19_39_29__09_04_2024"
    )