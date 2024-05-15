"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports

import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object,
)
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt


def plot_distributions(
    fileName, M_networks, scenarios_seeds_titles, network_titles, colors_scenarios, fixed_emissions
):

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        for i, seed_type in enumerate(scenarios_seeds_titles):
            ax.hist(emissions[i], bins=30, alpha=0.5, label= seed_type, color = colors_scenarios[i])
        
        #ax.hist(fixed_emissions, bins=30, alpha=1, label= "Fixed preferences", color = "black")
        ax.vlines(x = fixed_emissions, ymin = 0, ymax= 10, label= "Fixed preferences", linestyles="--", color= "black", )
        ax.set_xlabel(r"Cumulative Emisisons, E")
        ax.set_title (network_titles[k])
        ax.set_ylim(0,120)

    axes[0].set_ylabel(r"Frequency")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_seeds_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_hist_init_seed"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_distributions_rows(fileName, M_networks, scenarios_seeds_titles, network_titles, colors_scenarios, fixed_emissions, base_params):
    num_networks = len(M_networks)
    num_scenarios = len(scenarios_seeds_titles)
    
    fig, axes = plt.subplots(ncols=num_networks, nrows=num_scenarios, figsize=(20, 10), sharex=True, sharey=True,constrained_layout = True)
    
    y_max=400

    for i in range(num_scenarios):
        for j in range(num_networks):
            emissions = M_networks[j]
            ax = axes[i, j]

            a = ax.hist(emissions[i], bins=30, alpha=0.5, label=network_titles[j], color=colors_scenarios[i])
            ax.vlines(x=fixed_emissions, ymin=0, ymax=y_max, label="Fixed preferences", linestyles="--", color="black")

            #if i == 2:
            #    print(a)
            #   quit()

            ax.set_ylim(0, y_max)

            if j == 0:
                ax.set_ylabel(scenarios_seeds_titles[i])
                
            if i == 0:
                axes[0][j].set_title (network_titles[j])
                #ax.legend(loc='upper right', fontsize="10")
    fig.supxlabel(r"Cumulative Emissions, E")
    fig.supylabel(r"Frequency")
    fig.suptitle("Social susceptability, $\phi$ = " +  str(base_params["phi_lower"]) + ", Preference cohesion, $c$ = " +  str(base_params["coherance_state"]) )
    #fig.text(0.5, 0.04, 'Frequency', ha='center')
    #fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    
    
    #fig.tight_layout()
    
    plotName = fileName + "/Plots"
    f = plotName + "/network_hist_init_seed_rows"
    fig.savefig(f+ ".png", dpi=600, format="png") 


def main(
    fileName
) -> None:
    
    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    fixed_emissions = load_object(fileName + "/Data","fixed_emissions")
    print("FIXED EMISSIONS:", fixed_emissions)
    emissions_SW = load_object(fileName + "/Data","emissions_SW")
    emissions_SBM = load_object(fileName + "/Data","emissions_SBM")
    emissions_BA = load_object(fileName + "/Data","emissions_BA")

    emissions_networks = np.asarray([emissions_SW,emissions_SBM,emissions_BA])
    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"] 
    base_params = load_object(fileName + "/Data", "base_params") 
    scenarios_seed = ["Preferences", "Network links", "Neighbour shuffling", "Preference shuffling"]#load_object(fileName + "/Data", "scenarios_seed")
    #scenarios_seed = ["Environmental identity", "Network links"]
    #plot_distributions(fileName, emissions_networks, scenarios_seed, network_titles, colors_scenarios, fixed_emissions)
    plot_distributions_rows(fileName, emissions_networks, scenarios_seed, network_titles, colors_scenarios, fixed_emissions, base_params)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/init_seed_effect_gen_11_19_40__15_05_2024",
    )