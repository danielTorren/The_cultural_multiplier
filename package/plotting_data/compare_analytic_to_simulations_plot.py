"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports

from matplotlib.lines import lineStyles
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object,
    save_object,
    calc_bounds
)
from package.generating_data.static_preferences_emissions_gen import calc_required_static_carbon_tax_seeds
from package.plotting_data.price_elasticity_plot import calculate_price_elasticity,calc_price_elasticities_2D
from matplotlib.cm import rainbow,  get_cmap
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline,UnivariateSpline

def plot_sector_rebound_simulations(
    fileName, total_emissions_1, total_emissions_2,  emissions_networks_2, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,6), constrained_layout = True)

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions_1  = total_emissions_1[k].T
        emissions_2  = total_emissions_2[k].T
        emissiosn_sector_2 = np.transpose(emissions_networks_2[k],(2,1,0))

        
        emissions_1_mean = (emissions_1).mean(axis=0)
        emissions_2_mean = (emissions_2).mean(axis=0)
        emissiosn_sector_2_1_mean = (emissiosn_sector_2[0]).mean(axis=0)
        emissiosn_sector_2_2_mean = (emissiosn_sector_2[1]).mean(axis=0)


        #PLOT MEANS
        ax.plot(property_vals, emissions_2_mean, alpha = 1, color = "Blue", label = r'Emissions both sectors, $E_F$')#BLUE emisisons both sectors 2 sector
        ax.plot(property_vals, emissions_1_mean, alpha = 1, color = "Red", label = r'Emissions no sector 2, $E^{*}_{F}$', ls="--")#RED Dashed emisisons both sectors 2 sector
            
        ax.plot(property_vals, emissiosn_sector_2_1_mean, alpha = 1, color = "Orange", label = r'Emissions sector 1, $E_{F,1}$')#ORANGE emssions from sector 1 of the 2 sector run
        ax.plot(property_vals, emissiosn_sector_2_2_mean, alpha = 1, color = "Green", label = r'Emissions sector 2, $E_{F,2}$')#Green emssions from sector 2 of the 2 sector run        

        for v in range(len(emissions_1)):#loop through seeds
            ax.plot(property_vals, emissions_2[v], alpha = 0.2, color = "Blue")#BLUE emisisons both sectors 2 sector
            ax.plot(property_vals, emissions_1[v], alpha = 0.2, color = "Red", ls="--")#RED Dashed emisisons both sectors 2 sector
            
            ax.plot(property_vals, emissiosn_sector_2[0][v], alpha = 0.2, color = "Orange")#ORANGE emssions from sector 1 of the 2 sector run
            ax.plot(property_vals, emissiosn_sector_2[1][v], alpha = 0.2, color = "Green")#Green emssions from sector 2 of the 2 sector run


        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")

    axes[2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_2_sector_rebound_simulations"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps") 


def main(
    fileName ,
) -> None:
    
    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list
    #print(colors)

    #quit()
    data_array_1 = load_object(fileName + "/Data", "data_array_1")
    data_array_2 = load_object(fileName + "/Data", "data_array_2")
    data_sectors_2 = load_object(fileName + "/Data", "data_sectors_2")

    
    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    scenario_labels = ["One sector, $M = 1$", "Two sectors, $M = 2$"]
    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 
    print("base params", base_params)

    #EMISSIONS PLOTS ALL TOGETHER SEEDS
    plot_sector_rebound_simulations(fileName, data_array_1, data_array_2, data_sectors_2,scenario_labels ,property_values_list,network_titles,colors_scenarios)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/compare_analytic_emissions_16_16_50__08_04_2024"
    )