"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports

from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object,
    calc_bounds
)
from matplotlib.cm import get_cmap


import matplotlib.pyplot as plt
import numpy as np

def plot_emissions_scatter(
    fileName, emissions, property_vals,
    colors_scenarios, scenario_labels
):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11, 5), sharey=True)

    colours_list = colors_scenarios[:len(emissions) * 2]
    colors_scenarios_complete = colours_list[0::2]

    for i in range(len(emissions)):
        Data = emissions[i]
        mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
        
        # Error bars for the confidence intervals
        error_bars = [mu_emissions - lower_bound, upper_bound - mu_emissions]
        
        # Scatter plot with error bars
        ax.errorbar(property_vals, mu_emissions, yerr=error_bars, fmt='o', color=colors_scenarios_complete[i], label=scenario_labels[i])

    ax.set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    ax.set_xlabel(r"Carbon price, $\tau$", fontsize="12")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize="9")
    fig.subplots_adjust(bottom=0.2)  # Adjust bottom margin to make space for legend

    plotName = fileName + "/Plots"
    f = plotName + "/emissions_scatter"
    fig.savefig(f + ".png", dpi=600, format="png")


def main(
    fileName,
) -> None:
    
    name = "tab20"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    #FULL
    Data_array = load_object(fileName + "/Data","Data_array")

    base_params = load_object(fileName + "/Data","base_params")
    a_preferences_vals = load_object(fileName + "/Data","a_preferences_vals")
    scenarios = load_object(fileName + "/Data","scenarios")

    #####################################################################################################

    scenario_labels = ["Fixed preferences","Dynamic social weighting", "Dynamic identity weighting"]  

    plot_emissions_scatter(fileName, Data_array,a_preferences_vals,colors_scenarios, scenario_labels)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/undershoot_15_56_02__20_08_2024",
       
    )