
# imports

import matplotlib.pyplot as plt
from package.resources.utility import (
    load_object,
    calc_bounds
)
from matplotlib.cm import get_cmap

def plot_emissions_confidence(
    fileName, emissions_networks,
    scenario_labels, property_vals, sub_vals,
    colors_scenarios
):

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(11, 5), sharey=True)

    for k, ax in enumerate(axes.flat):
        emissions = emissions_networks[k]

        colours_list = colors_scenarios[:len(emissions) * 2]
        colors_scenarios_complete = colours_list[0::2]

        for i in range(len(emissions)):
            Data = emissions[i]
            mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)

            # Plot the mean line
            ax.plot(property_vals, mu_emissions, label=scenario_labels[i], c=colors_scenarios_complete[i])

            # Plot the 95% confidence interval as a shaded area
            ax.fill_between(property_vals, lower_bound, upper_bound, color=colors_scenarios_complete[i], alpha=0.3)

        ax.set_title(rf"Substitutability, $\sigma$ = {sub_vals[k]}", fontsize=12)
        ax.grid()
    axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    axes[1].set_xlabel(r"Carbon tax, $\tau$", fontsize="12")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize="9")
    fig.subplots_adjust(bottom=0.17)  # Adjust bottom margin to make space for legend
    #plt.tight_layout()  # Adjust layout to make space for the legend
    
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_sub_confidence"
    fig.savefig(f + ".png", dpi=600, format="png")


def main(
    fileName
) -> None:
    
    name = "tab20"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    #FULL
    data_array = load_object(fileName + "/Data","data_array") 
    #print(data_array.shape)
    #quit()
    base_params = load_object(fileName + "/Data","base_params")

    #####################################################################################################

    network_titles = ["Small-World", "Stochastic Block Model", "Scale-Free"]
    scenario_labels = ["Fixed preferences","Social multiplier", "Cultural multiplier"]

    sub_vals = load_object(fileName + "/Data", "substitutability_vals")    
    property_values_list = load_object(fileName + "/Data", "property_values_list")       

    #plot_emissions_lines(fileName, emissions_networks, scenario_labels, property_values_list, network_titles,colors_scenarios)
    plot_emissions_confidence(fileName, data_array, scenario_labels, property_values_list, sub_vals,colors_scenarios)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/tax_sweep_networks_sub_11_07_48__29_10_2024"#tax_sweep_networks_sub_09_31_18__29_10_2024"#tax_sweep_networks_16_44_43__18_09_2024",#tax_sweep_networks_15_57_56__22_08_2024",
    )