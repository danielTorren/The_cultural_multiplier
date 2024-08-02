"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports

import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object,
    save_object,
    calc_bounds
)
from package.generating_data.static_preferences_emissions_gen import calc_required_static_carbon_tax_seeds
from package.plotting_data.price_elasticity_plot import calculate_price_elasticity,calc_price_elasticities_2D
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

def plot_means_end_points_emissions(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)

            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i])

            data_trans = Data.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i], alpha = 0.1)


        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_means_end_points_emissions_lines"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps")   

def plot_means_end_points_emissions_lines_inset(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]

        max_val_inset = 0.1
        inset_ax = axes[k].inset_axes([0.55, 0.53, 0.4, 0.45])
        index_min = np.where(property_vals > max_val_inset)[0][0]#get index where its more than 0.8 carbon tax, this way independent of number of reps as longas more than 3       

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)

            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i])
            inset_ax.plot(property_vals[:index_min], mu_emissions[:index_min], c = colors_scenarios[i])

            data_trans = Data.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i], alpha = 0.1)
                inset_ax.plot(property_vals[:index_min], data_trans[v][:index_min], c = colors_scenarios[i], alpha = 0.1)


        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_means_end_points_emissions_lines_inset"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps")   

def plot_emisisons_simple(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i])
            data_trans = Data.T
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i], alpha = 0.1)


        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")  

def plot_emisisons_simple_short(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles, colors_scenarios, value_min
):

    #print(c,emissions_final)
    index_min = np.where(property_vals> value_min)[0][0]

    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))
    
    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c = colors_scenarios[i])
            data_trans = Data.T
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color = colors_scenarios[i], alpha = 0.1)


        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        #ax.set_xlim(0.0,1)
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_short"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")  


#####################################################################################################
def plot_M_simple(
    fileName, M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        scen_mu_min = []
        scen_mu_max = []
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            scen_mu_min.append(min(mu_emissions))
            scen_mu_max.append(max(mu_emissions))
            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i+1])

            data_trans = Data.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i+1], alpha = 0.1)

        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
        ax.set_ylim(min(scen_mu_min)*1.1, max(scen_mu_max)*1.1)

    axes[0].set_ylabel(r"Carbon tax multiplier, M")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_M_simple"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_M_simple_short(
    fileName, M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios, value_min
):

    index_min = np.where(property_vals> value_min)[0][0]

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        scen_mu_min = []
        scen_mu_max = []
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            scen_mu_min.append(min(mu_emissions[index_min:]))
            scen_mu_max.append(max(mu_emissions[index_min:]))
            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c = colors_scenarios[i+1])

            data_trans = Data.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color = colors_scenarios[i+1], alpha = 0.1)

        #ax.set_xlim(0.1,1)
        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
        #ax.set_ylim(min(scen_mu_min)*1.1, max(scen_mu_max)*1.1)

    axes[0].set_ylabel(r"Carbon tax reduction, M")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_M_simple_short"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 


def plot_M_lines(
    fileName, M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        scen_mu_min = []
        scen_mu_max = []
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)

            scen_mu_min.append(min(mu_emissions))
            scen_mu_max.append(max(mu_emissions))

            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i+1])

            data_trans = Data.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i+1], alpha = 0.1)

        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
        #ax.set_ylim(min(scen_mu_min)*1.1, max(scen_mu_max)*1.1)

    axes[0].set_ylabel(r"Carbon tax multiplier, M")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_multiplier_lines"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_M_lines_inset(
    fileName, M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        
        max_val_inset = 0.1
        inset_ax = axes[k].inset_axes([0.55, 0.53, 0.4, 0.45])
        index_min = np.where(property_vals > max_val_inset)[0][0]#get index where its more than 0.8 carbon tax, this way independent of number of reps as longas more than 3       

        scen_mu_min = []
        scen_mu_max = []
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)

            scen_mu_min.append(min(mu_emissions))
            scen_mu_max.append(max(mu_emissions))

            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i+1])
            inset_ax.plot(property_vals[:index_min], mu_emissions[:index_min], c = colors_scenarios[i+1])

            data_trans = Data.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i+1], alpha = 0.1)
                inset_ax.plot(property_vals[:index_min], data_trans[v][:index_min], c = colors_scenarios[i+1], alpha = 0.1)

        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
        ax.set_ylim(min(scen_mu_min)*1.1, max(scen_mu_max)*1.1)

    axes[0].set_ylabel(r"Carbon tax multiplier, M")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_means_end_points_multiplier_lines_inset"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps")    

def plot_M_lines_short_inset(
    fileName, M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        
        max_val_inset = 0.08
        inset_ax = axes[k].inset_axes([0.55, 0.05, 0.4, 0.4])
        index_min = np.where(property_vals > max_val_inset)[0][0]#get index where its more than 0.8 carbon tax, this way independent of number of reps as longas more than 3       

        scen_mu_min = []
        scen_mu_max = []
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)

            scen_mu_min.append(min(mu_emissions))
            scen_mu_max.append(max(mu_emissions))

            ax.plot(property_vals[50:], mu_emissions[50:], label=scenarios_titles[i], c = colors_scenarios[i+1])
            inset_ax.plot(property_vals[:index_min], mu_emissions[:index_min], c = colors_scenarios[i+1])

            data_trans = Data.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals[50:], data_trans[v][50:], color = colors_scenarios[i+1], alpha = 0.1)
                inset_ax.plot(property_vals[:index_min], data_trans[v][:index_min], c = colors_scenarios[i+1], alpha = 0.1)

        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
        inset_ax.set_ylim(min(scen_mu_min)*1.1, max(scen_mu_max)*1.1)

    axes[0].set_ylabel(r"Carbon tax multiplier, M")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_means_end_points_multiplier_lines_inset_short"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps")    

def plot_M_lines_short(
    fileName, M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        
        #max_val_inset = 0.1
        #inset_ax = axes[k].inset_axes([0.55, 0.53, 0.4, 0.45])
        #index_min = np.where(property_vals > max_val_inset)[0][0]#get index where its more than 0.8 carbon tax, this way independent of number of reps as longas more than 3       

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)

            #ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i])
            ax.plot(property_vals[50:], mu_emissions[50:], label=scenarios_titles[i], c = colors_scenarios[i+1])
            #inset_ax.plot(property_vals[:index_min], mu_emissions[:index_min], c = colors_scenarios[i])

            #data_trans = Data.T
            #quit()
            #for v in range(len(data_trans)):
            #    ax.plot(property_vals, data_trans[v], color = colors_scenarios[i], alpha = 0.1)
            #    inset_ax.plot(property_vals[:index_min], data_trans[v][:index_min], c = colors_scenarios[i], alpha = 0.1)


        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Carbon tax multiplier, M")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_m_short"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps")    


def ALT_plot_M_simple_short_manual_colours(
    fileName_full,  M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios, value_min
):

    index_min = np.where(property_vals> value_min)[0][0]

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(11, 5), sharey=True)

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        scen_mu_min = []
        scen_mu_max = []

        colours_list = colors_scenarios[:(len(emissions)+1)*2]

        colors_scenarios_complete = colours_list[0::2]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]

            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)

            scen_mu_min.append(min(mu_emissions[index_min:]))
            scen_mu_max.append(max(mu_emissions[index_min:]))
            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c = colors_scenarios_complete[i+1])

            data_trans = Data.T

            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color = colors_scenarios_complete[i+1], alpha = 0.1)
                
        ax.set_ylim(0.8, 1.005)

        ax.set_title(network_titles[k], fontsize="12")

    axes[0].set_ylabel(r"Carbon price reduction, M", fontsize="12")
    axes[1].set_xlabel(r"Carbon price, $\tau$", fontsize="12")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize="9")
    fig.subplots_adjust(bottom=0.2)  # Adjust bottom margin to make space for legend
    #plt.tight_layout()  # Adjust layout to make space for the legend

    # plt.tight_layout()
    plotName = fileName_full + "/Plots"
    f = plotName + "/ALT_network_joint_M_simple_short_manual_colours"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps") 


def ALT_plot_emissions_simple_short_colours(
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
    f = plotName + "/ALT_network_emissions_simple_short_joint_colours"
    fig.savefig(f + ".png", dpi=600, format="png")


def main(
    fileName ,#= "results/tax_sweep_11_29_20__28_09_2023"
) -> None:
    
    name = "tab20"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list
    #print(colors)

    #quit()
    emissions_SW = load_object(fileName + "/Data","emissions_SW")
    emissions_SBM = load_object(fileName + "/Data","emissions_SBM")
    emissions_BA = load_object(fileName + "/Data","emissions_BA")

    emissions_networks = np.asarray([emissions_SW,emissions_SBM,emissions_BA])
    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    scenario_labels = ["Fixed preferences","Uniform weighting","Static social weighting","Static identity weighting","Dynamic social weighting", "Dynamic identity weighting"]
    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 
    #print("base params", base_params)

    scenarios = load_object(fileName + "/Data", "scenarios")

    index_list_simple = [0,4,5]#GET THE INTERESTING STUFF!!
    emissions_networks_simple = np.asarray([[network[x] for x in index_list_simple] for network in  emissions_networks])     
    scenario_labels_simple = [scenario_labels[x] for x in index_list_simple]

    #plot_means_end_points_emissions_lines_inset(fileName, emissions_networks, scenario_labels ,property_values_list,network_titles,colors_scenarios)
    #plot_emisisons_simple(fileName, emissions_networks_simple, scenario_labels_simple ,property_values_list,network_titles,colors_scenarios)
    
    #plot_means_end_points_emissions(fileName, emissions_networks, scenario_labels ,property_values_list,network_titles,colors_scenarios)

    value_min = 0.1
    #plot_emisisons_simple_short(fileName, emissions_networks_simple, scenario_labels_simple ,property_values_list,network_titles,colors_scenarios, value_min)
    M_vals_networks = load_object(fileName + "/Data", "M_vals_networks")
    #print(scenarios)
    #"""
    #EMISSIONS PLOTS ALL TOGETHER SEEDS
    index_list_simple_M = [3,4]
    M_vals_networks_simple = np.asarray([[network[x] for x in index_list_simple_M] for network in M_vals_networks]) 
    
    #plot_M_simple(fileName,M_vals_networks_simple, scenario_labels_simple[1:], property_values_list,network_titles,colors_scenarios)
    #plot_M_simple_short(fileName,M_vals_networks_simple, scenario_labels_simple[1:], property_values_list,network_titles,colors_scenarios, value_min)
    
    #plot_M_lines(fileName, M_vals_networks, scenario_labels[1:] ,property_values_list,network_titles,colors_scenarios)
    
    #plot_M_lines_inset(fileName, M_vals_networks, scenario_labels[1:] ,property_values_list,network_titles,colors_scenarios)
    #plot_M_lines_short(fileName, M_vals_networks, scenario_labels[1:] ,property_values_list,network_titles,colors_scenarios)
    #plot_M_lines_short_inset(fileName, M_vals_networks, scenario_labels[1:] ,property_values_list,network_titles,colors_scenarios)


    ALT_plot_emissions_simple_short_colours(fileName,emissions_networks_simple, scenario_labels_simple,property_values_list,network_titles,colors_scenarios, value_min)
    ALT_plot_M_simple_short_manual_colours(fileName,  M_vals_networks_simple, scenario_labels_simple, property_values_list, network_titles, colors_scenarios, value_min)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/tax_sweep_networks_14_58_29__17_05_2024"#"results/tax_sweep_networks_14_58_29__17_05_2024"#"results/tax_sweep_networks_14_58_29__17_05_2024",#COMPLETE
        #fileName= "results/asym_tax_sweep_networks_17_31_30__10_05_2024",#ASYM
    )