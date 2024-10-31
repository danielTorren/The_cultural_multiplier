# imports

import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object,
    calc_bounds
)
from matplotlib.cm import get_cmap

def plot_multiplier_confidence_two(
    fileName_full,  M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):
    
    # First figure: only the first column
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(5, 5), sharey=True)

    emissions_0 = M_networks[0]
    colours_list_0 = colors_scenarios[:(len(emissions_0) + 1) * 2]
    colors_scenarios_complete_0 = colours_list_0[0::2]

    for i in range(len(emissions_0)):
        Data = emissions_0[i]
        mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)

        ax1.plot(property_vals, mu_emissions, label=scenarios_titles[i], c=colors_scenarios_complete_0[i + 1])
        ax1.fill_between(property_vals, lower_bound, upper_bound, color=colors_scenarios_complete_0[i + 1], alpha=0.3)
    
    ax1.grid()
    ax1.set_title(network_titles[0], fontsize="12")
    ax1.set_ylabel(r"Carbon tax reduction, $M_{tax}$", fontsize="12")
    ax1.set_xlabel(r"Carbon tax, $\tau$", fontsize="12")
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    fig1.legend(handles_1, labels_1, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize="9")
    fig1.subplots_adjust(bottom=0.2)  # Adjust bottom margin for legend

    # Saving the first figure
    plotName = fileName_full + "/Plots"
    f1 = plotName + "/multiplier_conf_column1"
    fig1.savefig(f1 + ".png", dpi=600, format="png")
    fig1.savefig(f1 + ".eps", dpi=600, format="eps")

    # Second figure: second and third columns
    fig2, axes2 = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), sharey=True)

    for k, ax in enumerate(axes2.flat, 1):  # Start k from 1 for second and third columns
        emissions = M_networks[k]

        colours_list = colors_scenarios[:(len(emissions) + 1) * 2]
        colors_scenarios_complete = colours_list[0::2]

        for i in range(len(emissions)):
            Data = emissions[i]
            mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)

            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c=colors_scenarios_complete[i + 1])
            ax.fill_between(property_vals, lower_bound, upper_bound, color=colors_scenarios_complete[i + 1], alpha=0.3)
        ax.grid()
        ax.set_title(network_titles[k], fontsize="12")

    axes2[0].set_xlabel(r"Carbon tax, $\tau$", fontsize="12")
    axes2[1].set_xlabel(r"Carbon tax, $\tau$", fontsize="12")
    axes2[0].set_ylabel(r"Carbon tax reduction, $M_{tax}$", fontsize="12")
    handles_2, labels_2 = axes2[0].get_legend_handles_labels()
    fig2.legend(handles_2, labels_2, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize="9")
    fig2.subplots_adjust(bottom=0.2)  # Adjust bottom margin for legend

    # Saving the second figure
    f2 = plotName + "/multiplier_conf_columns2_3"
    fig2.savefig(f2 + ".png", dpi=600, format="png")
    fig2.savefig(f2 + ".eps", dpi=600, format="eps")

def plot_emissions_confidence_two(
    fileName, emissions_networks,
    scenario_labels, property_vals, network_titles,
    colors_scenarios
):
    # First figure: only the first column
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(5, 5), sharey=True)

    emissions_0 = emissions_networks[0]
    colours_list_0 = colors_scenarios[:len(emissions_0) * 2]
    colors_scenarios_complete_0 = colours_list_0[0::2]

    for i in range(len(emissions_0)):
        Data = emissions_0[i]
        mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)

        # Plot the mean line
        ax1.plot(property_vals, mu_emissions, label=scenario_labels[i], c=colors_scenarios_complete_0[i])

        # Plot the 95% confidence interval as a shaded area
        ax1.fill_between(property_vals, lower_bound, upper_bound, color=colors_scenarios_complete_0[i], alpha=0.3)
        ax1.grid()
    ax1.set_title(network_titles[0], fontsize="12")
    ax1.set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    ax1.set_xlabel(r"Carbon tax, $\tau$", fontsize="12")

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    fig1.legend(handles_1, labels_1, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize="9")
    fig1.subplots_adjust(bottom=0.2)  # Adjust bottom margin to make space for legend

    # Save the first figure
    plotName = fileName + "/Plots"
    f1 = plotName + "/network_emissions_confidence_column1"
    fig1.savefig(f1 + ".png", dpi=600, format="png")

    # Second figure: second and third columns
    fig2, axes2 = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), sharey=True)

    for k, ax in enumerate(axes2.flat, 1):  # Start k from 1 for second and third columns
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

        ax.set_title(network_titles[k], fontsize="12")
        ax.grid()
    axes2[0].set_xlabel(r"Carbon tax, $\tau$", fontsize="12")
    axes2[1].set_xlabel(r"Carbon tax, $\tau$", fontsize="12")
    axes2[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    handles_2, labels_2 = axes2[0].get_legend_handles_labels()
    fig2.legend(handles_2, labels_2, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize="9")
    fig2.subplots_adjust(bottom=0.2)  # Adjust bottom margin for legend

    # Save the second figure
    f2 = plotName + "/network_emissions_confidence_columns2_3"
    fig2.savefig(f2 + ".png", dpi=600, format="png")


def main(
    fileName,
    MULTIPLIER = 1
) -> None:
    
    name = "tab20"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    emissions_SW = load_object(fileName + "/Data","emissions_SW")
    emissions_SBM = load_object(fileName + "/Data","emissions_SBM")
    emissions_SF = load_object(fileName + "/Data","emissions_SF")
    emissions_networks = np.asarray([emissions_SW,emissions_SBM,emissions_SF])    
    
    base_params = load_object(fileName + "/Data","base_params")
    
    #####################################################################################################

    network_titles = ["Small-World", "Stochastic Block Model", "Scale-Free"]
    scenario_labels = ["Fixed preferences","Social multiplier", "Cultural multiplier"]

    property_values_list = load_object(fileName + "/Data", "property_values_list")       

    plot_emissions_confidence_two(fileName, emissions_networks, scenario_labels, property_values_list, network_titles,colors_scenarios)

    if MULTIPLIER:
        list_M_networks = load_object(fileName + "/Data","list_M_networks")
        scenario_labels_M = ["Social multiplier", "Cultural multiplier"]
        
        plot_multiplier_confidence_two(fileName,list_M_networks, scenario_labels_M, property_values_list, network_titles, colors_scenarios)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/tax_sweep_networks_01_23_14__30_10_2024",#tax_sweep_networks_14_40_25__28_10_2024",#tax_sweep_networks_15_57_56__22_08_2024",#",#tax_sweep_networks_15_40_36__13_09_2024
        MULTIPLIER = 1
    )