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
from matplotlib.cm import rainbow,  get_cmap
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline,UnivariateSpline

def plot_scatter_end_points_emissions_scatter(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
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
                ax.scatter(property_vals,  data[j], label=scenarios_titles[i] if j == 0 else "", c = colors_scenarios[i])

        
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
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,6), constrained_layout = True)

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, min_emissions, max_emissions = calc_bounds(Data, 0.95)
            #mu_emissions =  Data.mean(axis=1)
            #min_emissions =  Data.min(axis=1)
            #max_emissions=  Data.max(axis=1)

            #print("mu_emissions",mu_emissions)
            #ax.plot(property_vals, mu_emissions, c= color, label=scenarios_titles[i])
            #ax.fill_between(property_vals, min_emissions, max_emissions, facecolor=color , alpha=0.4)
            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i])
            ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.4, facecolor = colors_scenarios[i])

        #ax.legend()
        ax.set_xlabel(r"Carbon price, $\tau$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    axes[2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_means_end_points_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_emissions_ratio_scatter(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):
    
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,6), constrained_layout = True, sharey=True)
    
    #print(c,emissions_final)
    #fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)
    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_no_tax))))
    for k, ax in enumerate(axes.flat):
        emissions_tax  = emissions_networks[k]#SHAPE : scenario, reps, seeds
        property_vals_0_index = np.where(property_vals==0)[0][0]
        emissions_no_tax =  emissions_tax[:,property_vals_0_index,:]

        ax.set_title(network_titles[k])
        for i in range(len(emissions_no_tax)):
            
            #color = next(colors)#set color for whole scenario?

            data_tax =  emissions_tax[i].T#its now seed then tax
            data_no_tax = emissions_no_tax[i]#which is seed
            reshape_data_no_tax = data_no_tax[:, np.newaxis]

            data_ratio = data_tax/reshape_data_no_tax# this is 2d array of ratio, where the rows are different seeds inside of which are different taxes

            #print("data",data)
            for j in range(len(data_ratio)):#loop over seeds
                ax.scatter(property_vals,  data_ratio[j], label=scenarios_titles[i] if j == 0 else "",c = colors_scenarios[i]) #, color = color
    
    axes[0].set_xlabel(r"Carbon price, $\tau$")
    axes[0].set_ylabel(r'Ratio of cumulative emissions relative to no carbon price')
    axes[2].legend( fontsize="8")
    #ax.set_title(r'Ratio of emissions relative to no carbon price')

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_ratio_scatter"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_emissions_ratio_line(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):
    
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,6), constrained_layout = True, sharey=True)
    #print(c,emissions_final)
    #fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)
    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_no_tax))))
    for k, ax in enumerate(axes.flat):
        emissions_tax  = emissions_networks[k]#SHAPE : scenario, reps, seeds
        property_vals_0_index = np.where(property_vals==0)[0][0]
        emissions_no_tax =  emissions_tax[:,property_vals_0_index,:]
        ax.set_title(network_titles[k])
        for i in range(len(emissions_no_tax)):
            
            #color = next(colors)#set color for whole scenario?

            data_tax =  emissions_tax[i].T#its now seed then tax
            data_no_tax = emissions_no_tax[i]#which is seed
            reshape_data_no_tax = data_no_tax[:, np.newaxis]

            data_ratio = data_tax/reshape_data_no_tax# this is 2d array of ratio, where the rows are different seeds inside of which are different taxes

            Data = data_ratio.T
            #print("Data", Data)
            mu_emissions =  Data.mean(axis=1)
            min_emissions =  Data.min(axis=1)
            max_emissions=  Data.max(axis=1)

            #print("mu_emissions",mu_emissions)
            #print(property_vals, mu_emissions)
            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i])#c= color
            ax.fill_between(property_vals, min_emissions, max_emissions , alpha=0.4, facecolor = colors_scenarios[i])#facecolor=color
    
    axes[0].set_xlabel(r"Carbon price, $\tau$")
    axes[0].set_ylabel(r'Ratio of cumulative emissions relative to no carbon price')
    axes[2].legend( fontsize="8")
    #ax.set_title(r'Ratio of emissions relative to no carbon price')

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_ratio_line"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_seeds_scatter_emissions(
    fileName, emissions_network, scenarios_titles, property_vals, seed_reps, seeds_to_show, network_titles, colors_scenarios
):
    
    fig, axes = plt.subplots(nrows=seeds_to_show, ncols= 3, figsize=(20,10), sharex=True, constrained_layout = True)
    
    emissions_trans = np.transpose(emissions_network,(0,3,1,2))#now its network, seed, scenario, reps
    #print("emissions_trans", emissions_trans.shape)

    for i, emissions_network_specific in enumerate(emissions_trans):
        emissions_reduc = emissions_network_specific[:seeds_to_show] # seed_reduc, scenario, reps
        axes[0][i].set_title(network_titles[i])
        for j in range(seeds_to_show):
            axes[j][0].set_ylabel("Seed = %s" % (j+1))
            axes[j,i].grid()  

            #colors = iter(rainbow(np.linspace(0, 1,len(scenarios_titles))))
            emissions = emissions_reduc[j]#this is a 2d array, scenarios and reps

            for k in range(len(emissions)):#cycle through scenarios
                
                #color = next(colors)#set color for whole scenario?
                data = emissions[k]#its now seed then tax
                #print("data",data)
                axes[j,i].scatter(property_vals,  data, label=scenarios_titles[k], s=8, c = colors_scenarios[k])

          
    fig.supxlabel(r"Carbon price, $\tau$")
    fig.supylabel(r"Cumulative carbon emissions, E")
    #axes[2].legend( fontsize="8")
    axes[0][-1].legend( fontsize="6")
    #print("what worong")
    plotName = fileName + "/Prints"
    f = plotName + "/seeds_scatter_carbon_tax_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_seeds_plot_emissions(
    fileName, emissions_network, scenarios_titles, property_vals, seed_reps, seeds_to_show, network_titles, colors_scenarios
):
    
    fig, axes = plt.subplots(nrows=seeds_to_show, ncols= 3, figsize=(20,10), sharex=True, constrained_layout = True)
    
    emissions_trans = np.transpose(emissions_network,(0,3,1,2))#now its network, seed, scenario, reps
    #print("emissions_trans", emissions_trans.shape)

    for i, emissions_network_specific in enumerate(emissions_trans):
        emissions_reduc = emissions_network_specific[:seeds_to_show] # seed_reduc, scenario, reps
        axes[0][i].set_title(network_titles[i])
        for j in range(seeds_to_show):
            axes[j][0].set_ylabel("Seed = %s" % (j+1))
            axes[j,i].grid()  

            #colors = iter(rainbow(np.linspace(0, 1,len(scenarios_titles))))
            emissions = emissions_reduc[j]#this is a 2d array, scenarios and reps

            for k in range(len(emissions)):#cycle through scenarios
                
                #color = next(colors)#set color for whole scenario?
                data = emissions[k]#its now seed then tax
                #print("data",data)
                axes[j,i].plot(property_vals,  data, label=scenarios_titles[k],c = colors_scenarios[k])

          
    fig.supxlabel(r"Carbon price, $\tau$")
    fig.supylabel(r"Cumulative carbon emissions, E")
    #axes[2].legend( fontsize="8")
    axes[0][-1].legend( fontsize="6")
    #print("what worong")
    plotName = fileName + "/Prints"
    f = plotName + "/seeds_line_carbon_tax_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_price_elasticies_seeds(
     fileName, emissions_network, scenarios_titles, property_vals, seed_reps, seeds_to_show, network_titles,colors_scenarios
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=seeds_to_show, ncols=3, figsize=(20, 10), constrained_layout=True)

    emissions_trans = np.transpose(emissions_network,(0,3,1,2))#now its network, seed, scenario, reps
    
    for i, emissions_network_specific in enumerate(emissions_trans):
        emissions_reduc = emissions_network_specific[:seeds_to_show] # seed_reduc, scenario, reps
        axes[0][i].set_title(network_titles[i])
        for j in range(seeds_to_show):
            axes[j][0].set_ylabel("Seed = %s" % (j+1))
            axes[j,i].grid()  

            #colors = iter(rainbow(np.linspace(0, 1,len(scenarios_titles))))
            emissions = emissions_reduc[j]#this is a 2d array, scenarios and reps

            for k in range(len(emissions)):#cycle through scenarios
                #DO CONVERSION
                data = emissions[k]
                price_elasticities = calculate_price_elasticity(property_vals,data)
                axes[j,i].plot(property_vals[1:], price_elasticities, label=scenarios_titles[k], c=colors_scenarios[k])
    
    fig.supxlabel(r"Carbon price, $\tau$")
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")
    axes[0][-1].legend( fontsize="6")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_price_elasticies_seeds"
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_price_elasticies_mean(fileName, emissions_network, scenarios_titles, property_vals, network_titles,colors_scenarios):

    fig, axes = plt.subplots(nrows= 1, ncols=3, figsize=(10, 6), constrained_layout=True)

    #print("scenarios_titles",scenarios_titles)

    for i, emissions_network_specific in enumerate(emissions_network):#(network, scenario, reps, seed)

        axes[i].set_title(network_titles[i])
        axes[i].grid() 

        for j,emissions_scenario in enumerate(emissions_network_specific):#cycle through scenarios
            #DO CONVERSION
            data_trans_full = []
            data_trans  = emissions_scenario.T
            for k in range(len(data_trans)):
                data_trans_full.append(calculate_price_elasticity(property_vals, data_trans[k]))
            Data = np.asarray(data_trans_full).T
            mu_emissions, min_emissions, max_emissions = calc_bounds(Data, 0.95)
            #mu_emissions =  Data.mean(axis=1)
            #min_emissions =  Data.min(axis=1)
            #max_emissions=  Data.max(axis=1)

            axes[i].plot(property_vals[1:], mu_emissions, label=scenarios_titles[j], c=colors_scenarios[j])#c= color
            axes[i].fill_between(property_vals[1:], min_emissions, max_emissions , alpha=0.4, facecolor=colors_scenarios[j])#facecolor=color
    
    fig.supxlabel(r"Carbon price, $\tau$")
    axes[0].set_ylabel(r"Price elasticity of emissions, $\epsilon$")
    axes[-1].legend( fontsize="6")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_price_elasticies_mean"
    fig.savefig(f + ".png", dpi=600, format="png")


def plot_emissions_ratio_seeds(fileName, emissions_network, scenarios_titles, property_vals, seed_reps, seeds_to_show, network_titles,colors_scenarios):

    fig, axes = plt.subplots(nrows=seeds_to_show, ncols=3, figsize=(20, 10), constrained_layout=True)
    #print("scenarios_titles",scenarios_titles)
    property_vals_fixed_preferences_index = scenarios_titles.index("Fixed preferences")
    #print("property_vals_fixed_preferences_index",property_vals_fixed_preferences_index)
    scenarios_titles_reduc =  scenarios_titles[:property_vals_fixed_preferences_index] + scenarios_titles[property_vals_fixed_preferences_index+1:] 
    #print("scenarios_titles_reduc",scenarios_titles_reduc)
    #quit()
    emissions_trans = np.transpose(emissions_network,(0,3,1,2))#now its network, seed, scenario, reps, before: network, scenario, reps, seed
    
    for i, emissions_network_specific in enumerate(emissions_trans):
        #print(i, property_vals_fixed_preferences_index)
        #print("emissions_network",emissions_network.shape)
        #quit()
        emissions_fixed_preferences_seed_reps = emissions_network[i,property_vals_fixed_preferences_index,:,:].T#its seed_to show, reps
        #quit()
        emissions_fixed_preferences_seed_to_show_reps = emissions_fixed_preferences_seed_reps[:seeds_to_show,:] 
        #print(emissions_fixed_preferences_seed_to_show_reps.shape)
        #quit()
        emissions_reduc = emissions_network_specific[:seeds_to_show,:] # seed_reduc, scenario, reps, THIS ACTUALL STILL INCLUDES THE FIXED SCENARIO

        axes[0][i].set_title(network_titles[i])
        for j in range(seeds_to_show):
            axes[j][0].set_ylabel("Seed = %s" % (j+1))
            axes[j,i].grid()  
            #colors = iter(rainbow(np.linspace(0, 1,len(scenarios_titles))))
            emissions_inc_fixed = emissions_reduc[j]#this is a 2d array, scenarios and reps
            #exlude the fixed preferenecs 
            emissions = np.delete(emissions_inc_fixed, property_vals_fixed_preferences_index, axis=0)
            #emissions_inc_fixed[:property_vals_fixed_preferences_index] + emissions_inc_fixed[property_vals_fixed_preferences_index+1:] 
        
            #print("emissions", emissions.shape)
            data_static_emissions = emissions_fixed_preferences_seed_to_show_reps[j]
            for k in range(len(emissions)):#cycle through scenarios
                #DO CONVERSION
                data = emissions[k] #this is the emissions data for a given netowrk,  seed and scenario
                #print(data)
                #print(data_static_emissions)
                #quit()
                emissions_ratio = data/data_static_emissions
                axes[j,i].plot(property_vals, emissions_ratio, label=scenarios_titles_reduc[k] , c=colors_scenarios[k+1])
    
    fig.supxlabel(r"Carbon price, $\tau$")
    fig.supylabel(r"Cumulative emissions ratio, $E/E_{\phi =0}$")
    axes[0][-1].legend( fontsize="6")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_ratio_seeds"
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_emissions_ratio_mean(fileName, emissions_network, scenarios_titles, property_vals, network_titles,colors_scenarios):

    fig, axes = plt.subplots(nrows= 1, ncols=3, figsize=(10, 6), constrained_layout=True)

    property_vals_fixed_preferences_index = scenarios_titles.index("Fixed preferences")

    #print("scenarios_titles",scenarios_titles)

    for i, emissions_network_specific in enumerate(emissions_network):#(network, scenario, reps, seed)

        emissions_fixed_preferences_seed_reps = emissions_network_specific[property_vals_fixed_preferences_index,:,:]#(scenario, reps, seed)

        emissions_network_specific_no_fixed = np.delete(emissions_network_specific, property_vals_fixed_preferences_index, axis=0)
        scenarios_titles_reduc = scenarios_titles[1:] 
        #print("scenarios_titles_reduc",scenarios_titles_reduc)
        #quit()
        axes[i].set_title(network_titles[i])
        axes[i].grid() 

        for j,emissions_scenario in enumerate(emissions_network_specific_no_fixed):#cycle through scenarios
            #DO CONVERSION
            Data = emissions_scenario/emissions_fixed_preferences_seed_reps[j]
            #Data = emissions_ratio.T

            mu_emissions =  Data.mean(axis=1)
            min_emissions =  Data.min(axis=1)
            max_emissions=  Data.max(axis=1)

            axes[i].plot(property_vals, mu_emissions, label=scenarios_titles_reduc[j], c=colors_scenarios[j+1])#c= color
            axes[i].fill_between(property_vals, min_emissions, max_emissions , alpha=0.4, facecolor=colors_scenarios[j+1])#facecolor=color
    
    fig.supxlabel(r"Carbon price, $\tau$")
    axes[0].set_ylabel(r"Cumulative emissions ratio, $E/E_{\phi =0}$")
    axes[-1].legend( fontsize="6")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_ratio_mean"
    fig.savefig(f + ".png", dpi=600, format="png")
    fig.savefig(f + ".eps", dpi=600, format="eps")


#####################################################################################################

def calc_M(tau_social,tau_static):

    return (1-tau_social/tau_static)#*100

def calc_M_vector(tau_social_vec ,tau_static_vec, emissions_social, emissions_static_full):
    #I WANT TO INTERPOLATE WHAT THE TAU VALUES OF THE STATIC ARE THAT GIVE THE SOCIAL

    #WHY DOES CUBIC SPLINES REQUIRE INCREASIGN VALUES??
    reverse_emissions_static_full = emissions_static_full[::-1]
    reverse_emissions_social = emissions_social[::-1]
    reverse_tau_static_vec = tau_static_vec[::-1]
    reverse_tau_social_vec = tau_social_vec[::-1]

    cs_static_input_emissions_output_tau = CubicSpline(reverse_emissions_static_full, reverse_tau_static_vec)
    predicted_reverse_tau_static = cs_static_input_emissions_output_tau(reverse_emissions_social)
    
    M_vals = calc_M(reverse_tau_social_vec[::-1],predicted_reverse_tau_static[::-1])

    return M_vals

def plot_reduc_seeds(fileName, emissions_network, scenarios_titles, property_vals,tau_matrix, emissions_matrix, seed_reps, seeds_to_show, network_titles,colors_scenarios):

    fig, axes = plt.subplots(nrows=seeds_to_show, ncols=3, figsize=(20, 10), constrained_layout=True)
    property_vals_fixed_preferences_index = scenarios_titles.index("Fixed preferences")
    scenarios_titles_reduc =  scenarios_titles[:property_vals_fixed_preferences_index] + scenarios_titles[property_vals_fixed_preferences_index+1:] 
    emissions_trans = np.transpose(emissions_network,(0,3,1,2))#now its network, seed, scenario, reps, before: network, scenario, reps, seed

    #print(" emissions_matrix shape" ,emissions_matrix.shape)
    #print("tau_matrix",tau_matrix.shape)
    for i, emissions_network_specific in enumerate(emissions_trans):
        #axes[i].set_ylim(-1,1)
        emissions_reduc = emissions_network_specific[:seeds_to_show,:] # seed_reduc, scenario, reps, THIS ACTUALL STILL INCLUDES THE FIXED SCENARIO

        axes[0][i].set_title(network_titles[i])
        for j in range(seeds_to_show):
            #axes[j][i].set_ylim(-1,1)
            axes[j][0].set_ylabel("Seed = %s" % (j+1))
            axes[j,i].grid()  
            #colors = iter(rainbow(np.linspace(0, 1,len(scenarios_titles))))
            emissions_inc_fixed = emissions_reduc[j]#this is a 2d array, scenarios and reps
            emissions = np.delete(emissions_inc_fixed, property_vals_fixed_preferences_index, axis=0)
            #print("social emissions.shape",emissions.shape)
            for k in range(len(emissions)):#cycle through scenarios
                #DO CONVERSION
                data = emissions[k] #this is the emissions data for a given netowrk,  seed and scenario
                tax_reduc = calc_M_vector(property_vals, tau_matrix[j], data, emissions_matrix[j])
                #print("tax_reduc",tax_reduc )
                #quit()
                axes[j,i].plot(property_vals, tax_reduc, label=scenarios_titles_reduc[k], c=colors_scenarios[k+1])
    
    fig.supxlabel(r"Carbon price, $\tau$")
    fig.supylabel(r"\% Carbon price reduction")
    axes[0][-1].legend( fontsize="6")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_tax_redu_seeds"
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_emissions_fixed(tau_matrix, emissions_matrix):
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.scatter(tau_matrix, emissions_matrix)

def plot_reduc_mean(fileName, emissions_network, scenarios_titles, property_vals,tau_matrix, emissions_matrix, seed_reps, network_titles,colors_scenarios):

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10), constrained_layout=True)

    property_vals_fixed_preferences_index = scenarios_titles.index("Fixed preferences")
    scenarios_titles_reduc =  scenarios_titles[:property_vals_fixed_preferences_index] + scenarios_titles[property_vals_fixed_preferences_index+1:] 
    emissions_trans = np.transpose(emissions_network,(0,3,1,2))#now its network, seed, scenario, reps, before: network, scenario, reps, seed

    for i, emissions_network_specific in enumerate(emissions_trans):
        axes[i].set_title(network_titles[i])
        #axes[i].set_ylim(-1,1)
        axes[i].grid()
        tax_reduc_matrix = []

        for j in range(seed_reps):
            #colors = iter(rainbow(np.linspace(0, 1,len(scenarios_titles))))
            emissions_inc_fixed = emissions_network_specific[j]#this is a 2d array, scenarios and reps
            emissions = np.delete(emissions_inc_fixed, property_vals_fixed_preferences_index, axis=0)

            data_row = []
            for k in range(len(emissions)):#cycle through scenarios
                #DO CONVERSION
                data = emissions[k] #this is the emissions data for a given netowrk,  seed and scenario
                tax_reduc = calc_M_vector(property_vals, tau_matrix[j], data, emissions_matrix[j])
                data_row.append(tax_reduc)

            tax_reduc_matrix.append(data_row)
        #currently, seed, scenario, reps need to good scenarios, reps, seed
        tax_reduc_matrix = np.transpose(tax_reduc_matrix,(1,2,0))

        #MATRIX NEED TO BE SCENARIOS BY SEED by reps
        #CALC MEAN AN STUFFF
        for q in range(len(emissions)):
            Data = tax_reduc_matrix[q]
            mu_emissions =  Data.mean(axis=1)
            min_emissions =  Data.min(axis=1)
            max_emissions=  Data.max(axis=1)

            axes[i].plot(property_vals, mu_emissions, label=scenarios_titles_reduc[q],c=colors_scenarios[q+1])#c= color
            axes[i].fill_between(property_vals, min_emissions, max_emissions , alpha=0.4)#facecolor=color
    
    fig.supxlabel(r"Carbon price, $\tau$")
    fig.supylabel(r"\% Carbon price reduction")
    axes[-1].legend( fontsize="10")

    

    plotName = fileName + "/Plots"
    f = plotName + "/plot_tax_redu_mean"
    fig.savefig(f + ".png", dpi=600, format="png")
    
def main(
    fileName ,#= "results/tax_sweep_11_29_20__28_09_2023"
    LOAD_STATIC_FULL = 0
) -> None:
    
    name = "Set2"
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
    print("base params", base_params)

    scenarios = load_object(fileName + "/Data", "scenarios")
    #print(scenarios)
    #"""
    #EMISSIONS PLOTS ALL TOGETHER SEEDS
    #plot_scatter_end_points_emissions_scatter(fileName, emissions_networks, scenario_labels ,property_values_list,network_titles,colors_scenarios)
    plot_means_end_points_emissions(fileName, emissions_networks, scenario_labels ,property_values_list,network_titles,colors_scenarios)

    #EMISSIONS RATIOS ALL TOGETHER, THIS IS THE RATIO OF EMISSIONS TO THE CASE OF NO CARBON PRICE
    #plot_emissions_ratio_scatter(fileName, emissions_networks, scenario_labels ,property_values_list,network_titles,colors_scenarios)
    #plot_emissions_ratio_line(fileName, emissions_networks, scenario_labels ,property_values_list,network_titles,colors_scenarios)
    
    #SEED EMISSIOSN PLOTS
    #seed_reps = base_params["seed_reps"]
    #seeds_to_show = 3
    #plot_seeds_scatter_emissions(fileName, emissions_networks, scenario_labels ,property_values_list,seed_reps,seeds_to_show,network_titles,colors_scenarios)
    #plot_seeds_plot_emissions(fileName, emissions_networks, scenario_labels ,property_values_list,seed_reps,seeds_to_show,network_titles,colors_scenarios)

    #PRICE ELASTICITIES
    #plot_price_elasticies_seeds(fileName, emissions_networks, scenario_labels,property_values_list,seed_reps,seeds_to_show,network_titles,colors_scenarios)
    #plot_price_elasticies_mean(fileName, emissions_networks, scenario_labels ,property_values_list,network_titles,colors_scenarios)

    #PLOT MULTIPLIER RELATIVE TO THE CASE OF PRICE EFFECT
    #plot_emissions_ratio_seeds(fileName, emissions_networks, scenario_labels ,property_values_list,seed_reps,seeds_to_show,network_titles,colors_scenarios)
    #plot_emissions_ratio_mean(fileName, emissions_networks, scenario_labels ,property_values_list,network_titles,colors_scenarios)
    
    
    #"""
    """
    seed_reps = base_params["seed_reps"]
    #seeds_to_show = 3

    if LOAD_STATIC_FULL:
        tau_matrix = load_object(fileName + "/Data", "tau_matrix")
        emissions_matrix = load_object(fileName + "/Data", "emissions_matrix")
    else:
        lower_bound, upper_bound = -0.8, 50#MIN AND MAX VALUES OF THE CARBON PRICE, NEGATIVE PRICE DOESNT HAVE CARBON DIVIDEND
        total_range_runs = 1000
        tau_matrix, emissions_matrix = calc_required_static_carbon_tax_seeds(base_params,property_values_list, emissions_networks,lower_bound, upper_bound, total_range_runs)#EMISSIONS CAN BE ANY OF THEM
        print("CALCULATED DATA")
        save_object(tau_matrix,fileName + "/Data", "tau_matrix")
        save_object(emissions_matrix,fileName + "/Data", "emissions_matrix")

    #plot_emissions_fixed(tau_matrix, emissions_matrix)
    #plot_reduc_seeds(fileName, emissions_networks, scenario_labels,property_values_list, tau_matrix, emissions_matrix,seed_reps,seeds_to_show,network_titles,colors_scenarios)
    plot_reduc_mean(fileName, emissions_networks, scenario_labels,property_values_list, tau_matrix, emissions_matrix,seed_reps,network_titles,colors_scenarios)
    """

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/tax_sweep_networks_10_49_39__06_04_2024",
        LOAD_STATIC_FULL = 0
    )