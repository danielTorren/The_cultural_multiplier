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



def plot_means_end_points_emissions_lines_inset(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10))#

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
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

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

    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

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

def plot_joint_emisisons_simple_short(
    fileName, fileName_asym, emissions_networks, emissions_networks_asym,  scenarios_titles, property_vals, network_titles, colors_scenarios, value_min
):

    #print(c,emissions_final)
    index_min = np.where(property_vals> value_min)[0][0]

    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))
    
    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]
        emissions_asym  = emissions_networks_asym[k]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            mu_emissions_asym, __, __ = calc_bounds(Data_asym, 0.95)
            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c = colors_scenarios[i])
            ax.plot(property_vals[index_min:], mu_emissions_asym[index_min:], c = colors_scenarios[i], linestyle = "--")
            
            data_trans = Data.T
            data_trans_asym = Data_asym.T
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color = colors_scenarios[i], alpha = 0.1)
                ax.plot(property_vals[index_min:], data_trans_asym[v][index_min:], color = colors_scenarios[i], alpha = 0.1, linestyle = "--")

        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        #ax.set_xlim(0.0,1)
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_short_joint"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")  

    # plt.tight_layout()
    plotName = fileName_asym + "/Plots"
    f = plotName + "/network_emissions_simple_short_joint"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")  

def plot_joint_emisisons_simple_short_colours(fileName, fileName_asym, emissions_networks, emissions_networks_asym,  scenarios_titles,  scenarios_titles_asym, property_vals, network_titles, colors_scenarios,colors_scenarios_asym, value_min
    ):

    index_min = np.where(property_vals> value_min)[0][0]

    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))
    
    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]
        emissions_asym  = emissions_networks_asym[k]

        colours_list = colors_scenarios[:len(emissions)*2]

        colors_scenarios_complete = colours_list[0::2]
        colors_scenarios_incomplete = colours_list[1::2]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            mu_emissions_asym, __, __ = calc_bounds(Data_asym, 0.95)

            colour = colors_scenarios[i]
            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c = colors_scenarios_complete[i])
            ax.plot(property_vals[index_min:], mu_emissions_asym[index_min:], label=scenarios_titles_asym[i], c = colors_scenarios_incomplete[i], linestyle = "--")
            
            data_trans = Data.T
            data_trans_asym = Data_asym.T
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color = colors_scenarios_complete[i], alpha = 0.1)
                ax.plot(property_vals[index_min:], data_trans_asym[v][index_min:], color = colors_scenarios_incomplete[i], alpha = 0.1, linestyle = "--")

        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        #ax.set_xlim(0.0,1)
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_short_joint_colours"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")  

    # plt.tight_layout()
    plotName = fileName_asym + "/Plots"
    f = plotName + "/network_emissions_simple_short_joint_colours"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps") 

def ALT_plot_joint_emissions_simple_short_colours(
    fileName, fileName_asym, emissions_networks, emissions_networks_asym,
    scenarios_titles, scenarios_titles_asym, property_vals, network_titles,
    colors_scenarios, colors_scenarios_asym, value_min
):
    index_min = np.where(property_vals > value_min)[0][0]

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(11, 5), sharey=True)

    for k, ax in enumerate(axes.flat):
        emissions = emissions_networks[k]
        emissions_asym = emissions_networks_asym[k]

        colours_list = colors_scenarios[:len(emissions) * 2]
        colors_scenarios_complete = colours_list[0::2]
        colors_scenarios_incomplete = colours_list[1::2]

        for i in range(len(emissions)):
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            mu_emissions, _, _ = calc_bounds(Data, 0.95)
            mu_emissions_asym, _, _ = calc_bounds(Data_asym, 0.95)

            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c=colors_scenarios_complete[i])
            ax.plot(property_vals[index_min:], mu_emissions_asym[index_min:], label=scenarios_titles_asym[i], c=colors_scenarios_incomplete[i], linestyle="--")

            data_trans = Data.T
            data_trans_asym = Data_asym.T
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color=colors_scenarios_complete[i], alpha=0.1)
                ax.plot(property_vals[index_min:], data_trans_asym[v][index_min:], color=colors_scenarios_incomplete[i], alpha=0.1, linestyle="--")

        ax.set_title(network_titles[k], fontsize="12")

    #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend

    
    axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    axes[1].set_xlabel(r"Carbon price, $\tau$", fontsize="12")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize="9")
    fig.subplots_adjust(bottom=0.2)  # Adjust bottom margin to make space for legend
    #plt.tight_layout()  # Adjust layout to make space for the legend
    
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_short_joint_colours"
    fig.savefig(f + ".png", dpi=600, format="png")

    plotName = fileName_asym + "/Plots"
    f = plotName + "/network_emissions_simple_short_joint_colours"
    fig.savefig(f + ".png", dpi=600, format="png")


def plot_joint_emisisons_colours(fileName, fileName_asym, emissions_networks, emissions_networks_asym,  scenarios_titles,  scenarios_titles_asym, property_vals, network_titles, colors_scenarios,colors_scenarios_asym, value_min
    ):

    index_min = np.where(property_vals> value_min)[0][0]

    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))
    
    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]
        emissions_asym  = emissions_networks_asym[k]

        colours_list = colors_scenarios[:len(emissions)*2]
        #print(len(colours_list), len(emissions)*2)
        #quit()


        colors_scenarios_complete = colours_list[0::2]
        colors_scenarios_incomplete = colours_list[1::2]
        #print(colors_scenarios_incomplete)
        #quit()
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            mu_emissions_asym, __, __ = calc_bounds(Data_asym, 0.95)

            #print(scenarios_titles[i])
            #print(colors_scenarios_complete[i])
            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios_complete[i])
            ax.plot(property_vals, mu_emissions_asym, label=scenarios_titles_asym[i], c = colors_scenarios_incomplete[i], linestyle = "--")
            
            data_trans = Data.T
            data_trans_asym = Data_asym.T
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios_complete[i], alpha = 0.1)
                ax.plot(property_vals, data_trans_asym[v], color = colors_scenarios_incomplete[i], alpha = 0.1, linestyle = "--")

        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        #ax.set_xlim(0.0,1)
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_joint_colours"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")  

    # plt.tight_layout()
    plotName = fileName_asym + "/Plots"
    f = plotName + "/network_emissions_joint_colours"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_joint_emisisons(
    fileName, fileName_asym, emissions_networks, emissions_networks_asym,  scenarios_titles, property_vals, network_titles, colors_scenarios, value_min
):

    #print(c,emissions_final)
    index_min = np.where(property_vals> value_min)[0][0]

    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))
    
    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]
        emissions_asym  = emissions_networks_asym[k]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            mu_emissions_asym, __, __ = calc_bounds(Data_asym, 0.95)
            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i])
            ax.plot(property_vals, mu_emissions_asym, c = colors_scenarios[i], linestyle = "--")
            
            data_trans = Data.T
            data_trans_asym = Data_asym.T
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i], alpha = 0.1)
                ax.plot(property_vals, data_trans_asym[v], color = colors_scenarios[i], alpha = 0.1, linestyle = "--")

        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        #ax.set_xlim(0.0,1)
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_joint"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")  

    # plt.tight_layout()
    plotName = fileName_asym + "/Plots"
    f = plotName + "/network_emissions_joint"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")

#####################################################################################################
def plot_M_simple(
    fileName, M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

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
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

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

def plot_joint_M_simple_short(
    fileName_full, fileName_asym,  M_networks, M_networks_asym, scenarios_titles, property_vals, network_titles, colors_scenarios, value_min
):

    index_min = np.where(property_vals> value_min)[0][0]

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        emissions_asym  = M_networks_asym[k]
        scen_mu_min = []
        scen_mu_max = []
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            mu_emissions_asym, __, __ = calc_bounds(Data_asym, 0.95)
            scen_mu_min.append(min(mu_emissions[index_min:]))
            scen_mu_max.append(max(mu_emissions[index_min:]))
            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c = colors_scenarios[i+1])
            ax.plot(property_vals[index_min:], mu_emissions_asym[index_min:], c = colors_scenarios[i+1], linestyle = "--")

            data_trans = Data.T
            data_trans_asym = Data_asym.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color = colors_scenarios[i+1], alpha = 0.1)
                ax.plot(property_vals[index_min:], data_trans_asym[v][index_min:], color = colors_scenarios[i+1], alpha = 0.1, linestyle = "--")
        #ax.set_xlim(0.1,1)
        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
        #ax.set_ylim(min(scen_mu_min)*1.1, max(scen_mu_max)*1.1)

    axes[0].set_ylabel(r"Carbon tax reduction, M")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName_full + "/Plots"
    f = plotName + "/network_joint_M_simple_short"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

    plotName = fileName_asym + "/Plots"
    f = plotName + "/network_joint_M_simple_short"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_joint_M_simple_short_manual(
    fileName_full, fileName_asym,  M_networks, M_networks_asym, scenarios_titles, property_vals, network_titles, colors_scenarios, value_min
):

    index_min = np.where(property_vals> value_min)[0][0]

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        emissions_asym  = M_networks_asym[k]
        scen_mu_min = []
        scen_mu_max = []
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            mu_emissions_asym, __, __ = calc_bounds(Data_asym, 0.95)
            scen_mu_min.append(min([min(mu_emissions[index_min:]),min(mu_emissions_asym[index_min:])]))
            scen_mu_max.append(max([max(mu_emissions[index_min:]),max(mu_emissions_asym[index_min:])]))
            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c = colors_scenarios[i+1])
            ax.plot(property_vals[index_min:], mu_emissions_asym[index_min:], c = colors_scenarios[i+1], linestyle = "--")

            data_trans = Data.T
            data_trans_asym = Data_asym.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color = colors_scenarios[i+1], alpha = 0.1)
                ax.plot(property_vals[index_min:], data_trans_asym[v][index_min:], color = colors_scenarios[i+1], alpha = 0.1, linestyle = "--")
        #ax.set_xlim(0.1,1)
        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
        ax.set_ylim(0.8, 1.005)

    axes[0].set_ylabel(r"Carbon tax reduction, M")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName_full + "/Plots"
    f = plotName + "/network_joint_M_simple_short_manual"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

    plotName = fileName_asym + "/Plots"
    f = plotName + "/network_joint_M_simple_short_manual"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_joint_M_simple_short_manual_colours(
    fileName_full, fileName_asym,  M_networks, M_networks_asym, scenarios_titles,scenarios_titles_incomplete, property_vals, network_titles, colors_scenarios, value_min
):

    index_min = np.where(property_vals> value_min)[0][0]

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        emissions_asym  = M_networks_asym[k]
        scen_mu_min = []
        scen_mu_max = []

        colours_list = colors_scenarios[:(len(emissions)+1)*2]

        colors_scenarios_complete = colours_list[0::2]
        colors_scenarios_incomplete = colours_list[1::2]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            mu_emissions_asym, __, __ = calc_bounds(Data_asym, 0.95)
            scen_mu_min.append(min([min(mu_emissions[index_min:]),min(mu_emissions_asym[index_min:])]))
            scen_mu_max.append(max([max(mu_emissions[index_min:]),max(mu_emissions_asym[index_min:])]))
            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c = colors_scenarios_complete[i+1])
            ax.plot(property_vals[index_min:], mu_emissions_asym[index_min:], label=scenarios_titles_incomplete[i], c = colors_scenarios_incomplete[i+1], linestyle = "--")

            data_trans = Data.T
            data_trans_asym = Data_asym.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color = colors_scenarios_complete[i+1], alpha = 0.1)
                ax.plot(property_vals[index_min:], data_trans_asym[v][index_min:], color = colors_scenarios_incomplete[i+1], alpha = 0.1, linestyle = "--")
        #ax.set_xlim(0.1,1)
        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
        ax.set_ylim(0.8, 1.005)

    axes[0].set_ylabel(r"Carbon tax reduction, M")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName_full + "/Plots"
    f = plotName + "/network_joint_M_simple_short_manual_colours"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

    plotName = fileName_asym + "/Plots"
    f = plotName + "/network_joint_M_simple_short_manual_colours"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 


def ALT_plot_joint_M_simple_short_manual_colours(
    fileName_full, fileName_asym,  M_networks, M_networks_asym, scenarios_titles,scenarios_titles_incomplete, property_vals, network_titles, colors_scenarios, value_min
):

    index_min = np.where(property_vals> value_min)[0][0]

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(11, 5), sharey=True)

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        emissions_asym  = M_networks_asym[k]
        scen_mu_min = []
        scen_mu_max = []

        colours_list = colors_scenarios[:(len(emissions)+1)*2]

        colors_scenarios_complete = colours_list[0::2]
        colors_scenarios_incomplete = colours_list[1::2]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            mu_emissions_asym, __, __ = calc_bounds(Data_asym, 0.95)
            scen_mu_min.append(min([min(mu_emissions[index_min:]),min(mu_emissions_asym[index_min:])]))
            scen_mu_max.append(max([max(mu_emissions[index_min:]),max(mu_emissions_asym[index_min:])]))
            ax.plot(property_vals[index_min:], mu_emissions[index_min:], label=scenarios_titles[i], c = colors_scenarios_complete[i+1])
            ax.plot(property_vals[index_min:], mu_emissions_asym[index_min:], label=scenarios_titles_incomplete[i], c = colors_scenarios_incomplete[i+1], linestyle = "--")

            data_trans = Data.T
            data_trans_asym = Data_asym.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals[index_min:], data_trans[v][index_min:], color = colors_scenarios_complete[i+1], alpha = 0.1)
                ax.plot(property_vals[index_min:], data_trans_asym[v][index_min:], color = colors_scenarios_incomplete[i+1], alpha = 0.1, linestyle = "--")
        #ax.set_xlim(0.1,1)
        #ax.legend()
        #ax.set_xlabel(r"Carbon price, $\tau$")
        #ax.set_title (network_titles[k])
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
    f = plotName + "/alt_network_joint_M_simple_short_manual_colours"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps") 

    plotName = fileName_asym + "/Plots"
    f = plotName + "/alt_network_joint_M_simple_short_manual_colours"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps") 



def plot_joint_M_colours(
    fileName_full, fileName_asym,  M_networks, M_networks_asym, scenarios_titles,scenarios_titles_incomplete, property_vals, network_titles, colors_scenarios, value_min
):

    index_min = np.where(property_vals> value_min)[0][0]

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        emissions_asym  = M_networks_asym[k]
        scen_mu_min = []
        scen_mu_max = []

        colours_list = colors_scenarios[:(len(emissions)+1)*2]

        colors_scenarios_complete = colours_list[0::2]
        colors_scenarios_incomplete = colours_list[1::2]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            mu_emissions_asym, __, __ = calc_bounds(Data_asym, 0.95)
            scen_mu_min.append(min([min(mu_emissions),min(mu_emissions_asym)]))
            scen_mu_max.append(max([max(mu_emissions),max(mu_emissions_asym)]))
            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios_complete[i+1])
            ax.plot(property_vals, mu_emissions_asym, label=scenarios_titles_incomplete[i], c = colors_scenarios_incomplete[i+1], linestyle = "--")

            data_trans = Data.T
            data_trans_asym = Data_asym.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios_complete[i+1], alpha = 0.1)
                ax.plot(property_vals, data_trans_asym[v], color = colors_scenarios_incomplete[i+1], alpha = 0.1, linestyle = "--")
        #ax.set_xlim(0.1,1)
        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
        ax.set_ylim(0.8, 1.005)

    axes[0].set_ylabel(r"Carbon tax reduction, M")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName_full + "/Plots"
    f = plotName + "/network_joint_M__colours"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

    plotName = fileName_asym + "/Plots"
    f = plotName + "/network_joint_colours"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 



def plot_joint_M(
    fileName_full, fileName_asym,  M_networks, M_networks_asym, scenarios_titles, property_vals, network_titles, colors_scenarios, value_min
):

    index_min = np.where(property_vals> value_min)[0][0]

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        emissions_asym  = M_networks_asym[k]
        scen_mu_min = []
        scen_mu_max = []
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            Data_asym = emissions_asym[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)
            mu_emissions_asym, __, __ = calc_bounds(Data_asym, 0.95)

            scen_mu_min.append(min(mu_emissions[index_min:]))
            scen_mu_max.append(max(mu_emissions[index_min:]))

            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i+1])
            ax.plot(property_vals, mu_emissions_asym, c = colors_scenarios[i+1], linestyle = "--")

            data_trans = Data.T
            data_trans_asym = Data_asym.T
            #quit()
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i+1], alpha = 0.1)
                ax.plot(property_vals, data_trans_asym[v], color = colors_scenarios[i+1], alpha = 0.1, linestyle = "--")
        #ax.set_xlim(0.1,1)
        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        ax.set_title (network_titles[k])
        ax.set_ylim(min(scen_mu_min)*1.1, max(scen_mu_max)*1.1)

    axes[0].set_ylabel(r"Carbon tax reduction, M")
    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName_full + "/Plots"
    f = plotName + "/network_joint_M"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

    plotName = fileName_asym + "/Plots"
    f = plotName + "/network_joint_M"
    fig1.savefig(f+ ".png", dpi=600, format="png") 
    fig1.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_M_lines(
    fileName, M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10))#

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
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10))#

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
    fig1, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10))#

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
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,10))#

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

def main(
    fileName_full,  
    fileName_asym
) -> None:

    colors_scenarios_asym = get_cmap("tab20").colors
    #print(colors)

    #FULL
    emissions_SW_full = load_object(fileName_full + "/Data","emissions_SW")
    emissions_SBM_full = load_object(fileName_full + "/Data","emissions_SBM")
    emissions_BA_full = load_object(fileName_full + "/Data","emissions_BA")
    emissions_networks_full = np.asarray([emissions_SW_full,emissions_SBM_full,emissions_BA_full])
    M_vals_networks_full = load_object(fileName_full + "/Data", "M_vals_networks")
    
    #ASYM
    emissions_SW_asym = load_object(fileName_asym + "/Data","emissions_SW")
    emissions_SBM_asym = load_object(fileName_asym + "/Data","emissions_SBM")
    emissions_BA_asym = load_object(fileName_asym + "/Data","emissions_BA")
    emissions_networks_asym = np.asarray([emissions_SW_asym,emissions_SBM_asym,emissions_BA_asym])
    M_vals_networks_asym = load_object(fileName_asym + "/Data", "M_vals_networks")

    #####################################################################################################

    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    scenario_labels = ["Fixed preferences","Uniform weighting","Static social weighting","Static identity weighting","Dynamic social weighting", "Dynamic identity weighting"]
    property_values_list = load_object(fileName_full + "/Data", "property_values_list")       
    base_params_full = load_object(fileName_full + "/Data", "base_params") 
    base_params_asym = load_object(fileName_asym + "/Data", "base_params") 
    scenarios = load_object(fileName_full + "/Data", "scenarios")

    #####################################################################################################
    #SIMPLE
    index_list_simple = [0,4,5]#GET THE INTERESTING STUFF!!
    emissions_networks_simple_full = np.asarray([[network[x] for x in index_list_simple] for network in  emissions_networks_full])     
    emissions_networks_simple_asym = np.asarray([[network[x] for x in index_list_simple] for network in  emissions_networks_asym]) 
    scenario_labels_simple = [scenario_labels[x] for x in index_list_simple]

    index_list_simple_M = [3,4]
    M_vals_networks_simple_full = np.asarray([[network[x] for x in index_list_simple_M] for network in M_vals_networks_full]) 
    M_vals_networks_simple_asym = np.asarray([[network[x] for x in index_list_simple_M] for network in M_vals_networks_asym]) 
    value_min = 0.1

    #####################################################################################################
    #SIMPLE plots
    #plot_joint_emisisons_simple_short(fileName_full, fileName_asym, emissions_networks_simple_full, emissions_networks_simple_asym, scenario_labels_simple ,property_values_list,network_titles,colors_scenarios, value_min)
    #plot_joint_M_simple_short(fileName_full, fileName_asym, M_vals_networks_simple_full, M_vals_networks_simple_asym, scenario_labels_simple[1:], property_values_list,network_titles,colors_scenarios, value_min)
    #plot_joint_M_simple_short_manual(fileName_full, fileName_asym, M_vals_networks_simple_full, M_vals_networks_simple_asym, scenario_labels_simple[1:], property_values_list,network_titles,colors_scenarios, value_min)
    #ALL THE COMPLEXITY
    #plot_joint_emisisons(fileName_full, fileName_asym, emissions_networks_full, emissions_networks_asym, scenario_labels ,property_values_list,network_titles,colors_scenarios, value_min)
    #plot_joint_M(fileName_full, fileName_asym, M_vals_networks_simple_full, M_vals_networks_simple_asym, scenario_labels[1:], property_values_list,network_titles,colors_scenarios, value_min)


    #SEPERATING OUT THE COLOURS
    name = "tab20"#"tab20"#"Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    scenario_labels_simple_total = ["Complete coverage fixed preferences", "Complete coverage social multiplier", "Complete coverage environmental identity multiplier"]
    scenario_labels_simple_asym = ["Incomplete coverage fixed preferences", "Incomplete coverage social multiplier", "Incomplete coverage environmental identity multiplier"]
    #plot_joint_emisisons_simple_short_colours(fileName_full, fileName_asym, emissions_networks_simple_full, emissions_networks_simple_asym, scenario_labels_simple_total, scenario_labels_simple_asym ,property_values_list,network_titles,colors_scenarios,colors_scenarios_asym, value_min)
    ALT_plot_joint_emissions_simple_short_colours(fileName_full, fileName_asym, emissions_networks_simple_full, emissions_networks_simple_asym, scenario_labels_simple_total, scenario_labels_simple_asym ,property_values_list,network_titles,colors_scenarios,colors_scenarios_asym, value_min)
    #ALT_plot_joint_M_simple_short_manual_colours(fileName_full, fileName_asym, M_vals_networks_simple_full, M_vals_networks_simple_asym,  scenario_labels_simple_total[1:], scenario_labels_simple_asym[1:], property_values_list,network_titles,colors_scenarios, value_min)
    
    ##############FULLL
    name = "tab20"#"tab20"#"Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list
    scenario_labels_complete = ["Complete coverage fixed preferences","Complete coverage uniform weighting","Complete coverage static social multiplier","Complete coverage static environmental identity multiplier","Complete coverage social multiplier", "Complete coverage environmental identity multiplier"]
    scenario_labels_incomplete = ["Incomplete coverage fixed preferences","Incomplete coverage uniform weighting","Incomplete coverage static social multiplier","Incomplete coverage static environmental identity multiplier","Incomplete coverage social multiplier", "Incomplete coverage environmental identity multiplier"]

    #plot_joint_emisisons_colours(fileName_full, fileName_asym, emissions_networks_full, emissions_networks_asym, scenario_labels_complete, scenario_labels_incomplete ,property_values_list,network_titles,colors_scenarios,colors_scenarios_asym, value_min)
    #plot_joint_M_colours(fileName_full, fileName_asym, M_vals_networks_full, M_vals_networks_asym,  scenario_labels_complete[1:], scenario_labels_incomplete[1:] , property_values_list,network_titles,colors_scenarios, value_min)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName_full = "results/tax_sweep_networks_14_58_29__17_05_2024",#"results/tax_sweep_networks_11_10_59__14_06_2024",#tax_sweep_networks_13_18_41__19_04_2024",#FULL
        fileName_asym = "results/asym_tax_sweep_networks_13_09_20__20_05_2024"#"results/asym_tax_sweep_networks_11_11_23__14_06_2024"#asym_tax_sweep_networks_17_31_30__10_05_2024,#ASYM
    )