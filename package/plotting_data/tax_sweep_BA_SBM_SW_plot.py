"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object,
    save_object
)
from matplotlib.cm import rainbow
import matplotlib.pyplot as plt

def plot_scatter_end_points_emissions_scatter(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles
):

    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,6), constrained_layout = True, sharey=True)

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]
        for i in range(len(emissions)):
            
            #color = next(colors)#set color for whole scenario?
            data = emissions[i].T#its now seed then tax
            #print("data",data)
            for j in range(len(data)):
                #ax.scatter(property_vals,  data[j], color = color, label=scenarios_titles[i] if j == 0 else "")
                ax.scatter(property_vals,  data[j], label=scenarios_titles[i] if j == 0 else "")

        
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_ylabel(r"Cumulative carbon emissions, E")
        ax.set_title (network_titles[k])

    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    axes[2].legend(fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_scatter_carbon_tax_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_means_end_points_emissions(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,6), constrained_layout = True, sharey=True)

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions =  Data.mean(axis=1)
            min_emissions =  Data.min(axis=1)
            max_emissions=  Data.max(axis=1)

            #print("mu_emissions",mu_emissions)
            #ax.plot(property_vals, mu_emissions, c= color, label=scenarios_titles[i])
            #ax.fill_between(property_vals, min_emissions, max_emissions, facecolor=color , alpha=0.5)
            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i])
            ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)

        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    axes[2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_means_end_points_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def main(
    fileName = "results/tax_sweep_11_29_20__28_09_2023",
) -> None:
        
    emissions_SW = load_object(fileName + "/Data","emissions_SW")
    emissions_SBM = load_object(fileName + "/Data","emissions_SBM")
    emissions_BA = load_object(fileName + "/Data","emissions_BA")

    emissions_networks = [emissions_SW,emissions_SBM,emissions_BA]
    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    scenario_labels = ["Fixed preferences","Uniform weighting","Static social weighting","Static identity weighting","Dynamic social weighting", "Dynamic identity weighting"]
    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 
    #print("base params", base_params)
    scenarios = load_object(fileName + "/Data", "scenarios")
    print(scenarios)

    plot_scatter_end_points_emissions_scatter(fileName, emissions_networks, scenario_labels ,property_values_list,network_titles)
    plot_means_end_points_emissions(fileName, emissions_networks, scenario_labels ,property_values_list,network_titles)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/tax_sweep_networks_12_43_57__14_02_2024",
    )