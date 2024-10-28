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

def plot_emissions_confidence(
    fileName, emissions,
    scenario_labels, property_vals,
    colors_scenarios
):

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11, 5), sharey=True)


    colours_list = colors_scenarios[:len(emissions) * 2]
    colors_scenarios_complete = colours_list[0::2]

    for i in range(len(emissions)):
        Data = emissions[i]
        mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)

        # Plot the mean line
        ax.plot(property_vals, mu_emissions, label=scenario_labels[i], c=colors_scenarios_complete[i])

        # Plot the 95% confidence interval as a shaded area
        ax.fill_between(property_vals, lower_bound, upper_bound, color=colors_scenarios_complete[i], alpha=0.3)

    ax.set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    ax.set_xlabel(r"Carbon tax, $\tau$", fontsize="12")
    ax.legend()
    
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_confidence"
    fig.savefig(f + ".png", dpi=600, format="png")

def main(
    fileName,
    MULTIPLIER = 1
) -> None:
    
    name = "tab20"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    #FULL
    emissions_SW = load_object(fileName + "/Data","emissions_SW")
    
    base_params = load_object(fileName + "/Data","base_params")
    

    
    #####################################################################################################
    scenario_labels = ["Fixed preferences","Social multiplier", "Cultural multiplier", "Hybrid $25\%$",  "Hybrid $50\%$", "Hybrid $75\%$"]

    property_values_list = load_object(fileName + "/Data", "property_values_list")       

    
    plot_emissions_confidence(fileName, emissions_SW, scenario_labels, property_values_list,colors_scenarios)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/SW_tax_sweep_hybrid_21_21_59__28_10_2024",#tax_sweep_networks_15_57_56__22_08_2024",#",#tax_sweep_networks_15_40_36__13_09_2024
        MULTIPLIER = 0
    )