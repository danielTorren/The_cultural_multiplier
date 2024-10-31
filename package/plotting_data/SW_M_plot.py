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
    fileName, 
    fixed_emissions,
    socially_emissions,
    identity_emissions,
    scenario_labels, 
    property_vals, 
    colors_scenarios,
    carbon_tax_list  # Added carbon_tax_list as a parameter
):
    # Colors for scenarios (up to 3 scenarios)
    colours_list = colors_scenarios[:(3+1)*2]
    colors_scenarios_complete = colours_list[0::2]

    # Create figure with columns equal to the length of carbon_tax_list
    fig, axs = plt.subplots(ncols=len(carbon_tax_list), nrows=1, figsize=(5 * len(carbon_tax_list), 5), sharey=True)

    # Loop through each carbon tax level using carbon_tax_list
    for i, carbon_tax in enumerate(carbon_tax_list):
        # Get fixed emissions data for this carbon tax level and calculate bounds
        mu_emissions_fixed = np.mean(fixed_emissions,axis =1)

        mu_emissions_socially, lower_bound_socially, upper_bound_socially = calc_bounds(socially_emissions[i], 0.95)
        mu_emissions_identity, lower_bound_identity, upper_bound_identity = calc_bounds(identity_emissions[i], 0.95)

        # Plot fixed emissions with dashed line
        axs[i].axhline(y= mu_emissions_fixed[i], label=scenario_labels[0], c=colors_scenarios_complete[0], linestyle='--')

        # Plot socially informed emissions
        axs[i].plot(property_vals, mu_emissions_socially, label=scenario_labels[1], c=colors_scenarios_complete[1])
        axs[i].fill_between(property_vals, lower_bound_socially, upper_bound_socially, color=colors_scenarios_complete[1], alpha=0.3)

        # Plot identity-driven emissions
        axs[i].plot(property_vals, mu_emissions_identity, label=scenario_labels[2], c=colors_scenarios_complete[2])
        axs[i].fill_between(property_vals, lower_bound_identity, upper_bound_identity, color=colors_scenarios_complete[2], alpha=0.3)

        # Set title with carbon tax value
        axs[i].set_title(f"Carbon tax, $\\tau = {carbon_tax}$")
        axs[i].set_xlabel(r"Consumption categories, $M$", fontsize="12")

    # Set common y-axis label and add legend
    axs[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    axs[-1].legend()
    fig.legend(loc="upper right", bbox_to_anchor=(1.15, 1), fontsize="10")

    # Save the plot
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_confidence_M"
    fig.savefig(f + ".png", dpi=600, format="png")



def main(
    fileName,
) -> None:
    
    name = "tab20"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    #FULL
    emissions_array_socially = load_object(fileName + "/Data","emissions_array_socially")
    emissions_array_identity = load_object(fileName + "/Data","emissions_array_identity")

    fixed_emissions_array = load_object(fileName + "/Data","fixed_emissions_array")
    carbon_price_list = load_object(fileName + "/Data","carbon_price_list")
    scenario_labels = ["Fixed preferences","Social multiplier", "Cultural multiplier"]

    property_values_list = load_object(fileName + "/Data", "property_values_list")       

    plot_emissions_confidence(fileName, fixed_emissions_array,emissions_array_socially, emissions_array_identity, scenario_labels, property_values_list, colors_scenarios, carbon_price_list)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/M_vary_09_33_22__29_10_2024"
    )