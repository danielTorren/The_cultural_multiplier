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

def scenario_emissions_no_tax(
    fileName, emissions, scenarios_titles, seed_reps
):

    fig, ax = plt.subplots(figsize=(10,6),constrained_layout = True)

    data = emissions.T
    for i in range(len(data)):
        ax.scatter(scenarios_titles, data[i])
    ax.set_ylabel('Emissions stock, E')
    ax.set_title('No tax, emissions by Scenario')
    ax.set_xticks(np.arange(len(scenarios_titles)), scenarios_titles, rotation=45, ha='right')

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/scenario_emissions_no_tax"
    fig.savefig(f+ ".png", dpi=600, format="png")

def plot_scatter_end_points_emissions_scatter(
    fileName, emissions, scenarios_titles, property_vals
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)
    #print(len(emissions))
    colors = iter(rainbow(np.linspace(0, 1,len(emissions))))

    for i in range(len(emissions)):
        
        color = next(colors)#set color for whole scenario?
        data = emissions[i].T#its now seed then tax
        #print("data",data)
        for j in range(len(data)):
            ax.scatter(property_vals,  data[j], color = color, label=scenarios_titles[i] if j == 0 else "")

    ax.legend()
    ax.set_xlabel(r"Carbon Tax")
    ax.set_ylabel(r"Carbon Emissions")

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/scatter_carbon_tax_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_means_end_points_emissions(
    fileName, emissions, scenarios_titles, property_vals
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)

    colors = iter(rainbow(np.linspace(0, 1, len(emissions))))

    for i in range(len(emissions)):
        color = next(colors)#set color for whole scenario?
        Data = emissions[i]
        #print("Data", Data.shape)
        mu_emissions =  Data.mean(axis=1)
        min_emissions =  Data.min(axis=1)
        max_emissions=  Data.max(axis=1)

        #print("mu_emissions",mu_emissions)
        ax.plot(property_vals, mu_emissions, c= color, label=scenarios_titles[i])
        ax.fill_between(property_vals, min_emissions, max_emissions, facecolor=color , alpha=0.5)

    ax.legend()
    ax.set_xlabel(r"Carbon Tax")
    ax.set_ylabel(r"Carbon Emissions")

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/plot_means_end_points_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_emissions_ratio_scatter(
    fileName, emissions_no_tax, emissions_tax, scenarios_titles, property_vals
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)
    colors = iter(rainbow(np.linspace(0, 1,len(emissions_no_tax))))

    for i in range(len(emissions_no_tax)):
        
        color = next(colors)#set color for whole scenario?

        data_tax =  emissions_tax[i].T#its now seed then tax
        data_no_tax = emissions_no_tax[i]#which is seed
        reshape_data_no_tax = data_no_tax[:, np.newaxis]

        data_ratio = data_tax/reshape_data_no_tax# this is 2d array of ratio, where the rows are different seeds inside of which are different taxes

        #print("data",data)
        for j in range(len(data_ratio)):#loop over seeds
            ax.scatter(property_vals,  data_ratio[j], color = color, label=scenarios_titles[i] if j == 0 else "")

    ax.legend()
    ax.set_xlabel(r"Carbon Tax")
    ax.set_ylabel(r"Emissions ratio")
    ax.set_title(r'Ratio of taxed to no tax emissions stock by Scenario')

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_ratio_scatter"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_emissions_ratio_line(
    fileName, emissions_no_tax, emissions_tax, scenarios_titles, property_vals
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)
    colors = iter(rainbow(np.linspace(0, 1,len(emissions_no_tax))))

    for i in range(len(emissions_no_tax)):
        
        color = next(colors)#set color for whole scenario?

        data_tax =  emissions_tax[i].T#its now seed then tax
        data_no_tax = emissions_no_tax[i]#which is seed
        reshape_data_no_tax = data_no_tax[:, np.newaxis]

        data_ratio = data_tax/reshape_data_no_tax# this is 2d array of ratio, where the rows are different seeds inside of which are different taxes
        #print(data_ratio.shape)
        Data = data_ratio.T
        #print("Data", Data)
        mu_emissions =  Data.mean(axis=1)
        min_emissions =  Data.min(axis=1)
        max_emissions=  Data.max(axis=1)

        #print("mu_emissions",mu_emissions)
        #print(property_vals, mu_emissions)
        ax.plot(property_vals, mu_emissions, c= color, label=scenarios_titles[i])
        ax.fill_between(property_vals, min_emissions, max_emissions, facecolor=color , alpha=0.5)

    ax.legend()
    ax.set_xlabel(r"Carbon Tax")
    ax.set_ylabel(r"Emissions ratio")
    ax.set_title(r'Ratio of taxed to no tax emissions stock by Scenario')

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_ratio_line"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_emissions_ratio_scatter_alt(
    fileName, emissions_tax, scenarios_titles, property_vals
):
    
    """ get the index where carbon tax is equal to 0"""
    
    property_vals_0_index = np.where(property_vals==0)[0][0]
    emissions_no_tax =  emissions_tax[:,property_vals_0_index,:]

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)
    colors = iter(rainbow(np.linspace(0, 1,len(emissions_no_tax))))

    for i in range(len(emissions_no_tax)):
        
        color = next(colors)#set color for whole scenario?

        data_tax =  emissions_tax[i].T#its now seed then tax
        data_no_tax = emissions_no_tax[i]#which is seed
        reshape_data_no_tax = data_no_tax[:, np.newaxis]

        data_ratio = data_tax/reshape_data_no_tax# this is 2d array of ratio, where the rows are different seeds inside of which are different taxes

        #print("data",data)
        for j in range(len(data_ratio)):#loop over seeds
            ax.scatter(property_vals,  data_ratio[j], color = color, label=scenarios_titles[i] if j == 0 else "")

    ax.legend()
    ax.set_xlabel(r"Carbon Tax")
    ax.set_ylabel(r"Emissions ratio")
    ax.set_title(r'ALT Ratio of taxed to no tax emissions stock by Scenario')

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/ALT_plot_emissions_ratio_scatter"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_emissions_ratio_line_alt(
    fileName, emissions_tax, scenarios_titles, property_vals
):
    """ get the index where carbon tax is equal to 0"""
    
    property_vals_0_index = np.where(property_vals==0)[0][0]
    emissions_no_tax =  emissions_tax[:,property_vals_0_index,:]

    #emissions_no_tax = 

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout = True)
    colors = iter(rainbow(np.linspace(0, 1,len(emissions_no_tax))))

    for i in range(len(emissions_no_tax)):
        color = next(colors)#set color for whole scenario?

        data_tax =  emissions_tax[i].T#its now seed then tax
        data_no_tax = emissions_no_tax[i]#which is seed
        reshape_data_no_tax = data_no_tax[:, np.newaxis]

        data_ratio = data_tax/reshape_data_no_tax# this is 2d array of ratio, where the rows are different seeds inside of which are different taxes
        #print(data_ratio.shape)
        Data = data_ratio.T
        #print("Data", Data)
        mu_emissions =  Data.mean(axis=1)
        min_emissions =  Data.min(axis=1)
        max_emissions=  Data.max(axis=1)

        #print("mu_emissions",mu_emissions)
        #print(property_vals, mu_emissions)
        ax.plot(property_vals, mu_emissions, c= color, label=scenarios_titles[i])
        ax.fill_between(property_vals, min_emissions, max_emissions, facecolor=color , alpha=0.5)

    ax.legend()
    ax.set_xlabel(r"Carbon Tax")
    ax.set_ylabel(r"Emissions ratio")
    ax.set_title(r'ALT Ratio of taxed to no tax emissions stock by Scenario')

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/ALT_plot_emissions_ratio_line"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def plot_seeds_scatter_emissions(
    fileName, emissions_array, scenarios_titles, property_vals, seed_reps,seeds_to_show
):

    #print(c,emissions_final)
    #print(emissions.shape)#6,200,10 ie scenario, reps, seeds
    emissions_trans = np.transpose(emissions_array,(2,0,1))#now its seed, scenario, reps
    emissions_reduc = emissions_trans[:seeds_to_show]
    #quit()
    fig, axes = plt.subplots(ncols=seeds_to_show,figsize=(20,10),sharey=True, constrained_layout = True)


    for k, ax in enumerate(axes.flat):
        ax.grid()  
        #print("k",k)
        colors = iter(rainbow(np.linspace(0, 1,len(scenarios_titles))))
        emissions = emissions_reduc[k]#this is a 2d array
        #print(len(emissions))
        for i in range(len(emissions)):#cycle through scenarios
            
            color = next(colors)#set color for whole scenario?
            data = emissions[i]#its now seed then tax
            #print("data",data)
            ax.scatter(property_vals,  data, color = color, label=scenarios_titles[i])

          
        ax.set_xlabel(r"Carbon Tax")
        ax.set_ylabel(r"Carbon Emissions")
    axes[-1].legend()
    #print("what worong")
    plotName = fileName + "/Prints"
    f = plotName + "/seeds_scatter_carbon_tax_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def log_emissions_plot_culture(
    fileName, emissions_array, scenarios_titles, property_vals, seed_reps,seeds_to_show
):

    #print(c,emissions_final)
    #print(emissions.shape)#6,200,10 ie scenario, reps, seeds
    emissions_trans = np.transpose(emissions_array,(2,0,1))#now its seed, scenario, reps
    emissions_reduc = emissions_trans[:seeds_to_show]
    #quit()
    fig, axes = plt.subplots(ncols=seeds_to_show,figsize=(20,10),sharey=True, constrained_layout = True)


    for k, ax in enumerate(axes.flat):
        ax.grid()  
        #print("k",k)
        #colors = iter(rainbow(np.linspace(0, 1,len(scenarios_titles))))
        emissions = emissions_reduc[k]#this is a 2d array
        #print(len(emissions))
        #for i in range(len(emissions)):#cycle through scenarios
            
        #color = next(colors)#set color for whole scenario?
        data = emissions[5]#its now seed then tax
        #print("data",data)
        ax.scatter(property_vals,  data, label=scenarios_titles[5], color= "r")

        #ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_xlabel(r"Carbon Tax")
        ax.set_ylabel(r"Carbon Emissions")

    axes[-1].legend()
    #print("what worong")
    plotName = fileName + "/Prints"
    f = plotName + "/emissions_plot_culture"
    fig.savefig(f+ ".png", dpi=600, format="png")  

def main(
    fileName = "results/tax_sweep_11_29_20__28_09_2023",
) -> None:
        
    emissions_no_tax = load_object(fileName + "/Data","emissions_stock_no_tax")
    emissions_tax = load_object(fileName + "/Data","emissions_stock_tax")


    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 
    #print("base params", base_params)
    scenarios = load_object(fileName + "/Data", "scenarios")
    print(scenarios)

    seed_reps = base_params["seed_reps"]
    seeds_to_show = 4
    scenario_emissions_no_tax(fileName, emissions_no_tax, scenarios,seed_reps)
    plot_scatter_end_points_emissions_scatter(fileName, emissions_tax, scenarios ,property_values_list)
    plot_means_end_points_emissions(fileName, emissions_tax, scenarios ,property_values_list)
    plot_seeds_scatter_emissions(fileName, emissions_tax, scenarios ,property_values_list,seed_reps,seeds_to_show)
    #log_emissions_plot_culture(fileName, emissions_tax, scenarios ,property_values_list,seed_reps,seeds_to_show)
    #quit()
    
    #"""
    arr_zero_price = (np.where(property_values_list==0)[0])
    if arr_zero_price.size != 0:#check whether zero price included
        plot_emissions_ratio_scatter_alt(fileName, emissions_tax, scenarios ,property_values_list)
        plot_emissions_ratio_line_alt(fileName, emissions_tax, scenarios ,property_values_list)
        
    else:
        #if 0 price not include then divide by the zero tax, NOT really sure i need this could just get rid of zero tax entirely
        plot_emissions_ratio_scatter(fileName,emissions_no_tax, emissions_tax, scenarios ,property_values_list)
        plot_emissions_ratio_line(fileName,emissions_no_tax, emissions_tax, scenarios ,property_values_list)
    #"""
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/tax_sweep_13_36_35__28_11_2023",
    )