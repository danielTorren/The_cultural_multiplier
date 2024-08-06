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
from matplotlib.cm import get_cmap


def plot_emissions_lines(
    fileName, emissions_networks,
    scenarios_titles, property_vals, network_titles,
    colors_scenarios, value_min
):
    index_min = np.where(property_vals > value_min)[0][0]

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(11, 5), sharey=True)

    for k, ax in enumerate(axes.flat):
        emissions = emissions_networks[k]

        colours_list = colors_scenarios[:len(emissions) * 2]
        colors_scenarios_complete = colours_list[0::2]

        for i in range(len(emissions)):
            Data = emissions[i]
            mu_emissions, _, _ = calc_bounds(Data, 0.95)

            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c=colors_scenarios_complete[i])

            data_trans = Data.T
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color=colors_scenarios_complete[i], alpha=0.1)

        ax.set_title(network_titles[k], fontsize="12")
    
    axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    axes[1].set_xlabel(r"Carbon price, $\tau$", fontsize="12")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize="9")
    fig.subplots_adjust(bottom=0.2)  # Adjust bottom margin to make space for legend
    #plt.tight_layout()  # Adjust layout to make space for the legend
    
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_lines"
    fig.savefig(f + ".png", dpi=600, format="png")


def main(
    fileName,
) -> None:
    
    name = "tab20"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    #FULL
    emissions_SW = load_object(fileName + "/Data","emissions_SW")
    emissions_SBM = load_object(fileName + "/Data","emissions_SBM")
    emissions_BA = load_object(fileName + "/Data","emissions_BA")
    emissions_networks = np.asarray([emissions_SW,emissions_SBM,emissions_BA])    

    #####################################################################################################

    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    scenario_labels = ["Fixed preferences","Dynamic social weighting", "Dynamic identity weighting"]
    property_values_list = load_object(fileName + "/Data", "property_values_list")       

    value_min = 0
    plot_emissions_lines(fileName, emissions_networks, scenario_labels, property_values_list, network_titles,colors_scenarios, value_min)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/tax_sweep_networks_18_44_25__05_08_2024",
       
    )