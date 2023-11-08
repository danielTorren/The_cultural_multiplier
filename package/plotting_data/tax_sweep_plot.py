"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import (
    load_object
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

def main(
    fileName = "results/tax_sweep_11_29_20__28_09_2023",
) -> None:
        
    emissions_no_tax = load_object(fileName + "/Data","emissions_stock_no_tax")
    emissions_tax = load_object(fileName + "/Data","emissions_stock_tax")


    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 
    scenarios = load_object(fileName + "/Data", "scenarios")

    print("scenarios",scenarios)
    #print("emissions_no_tax",emissions_no_tax)
    #quit()
    #print("emissions_tax",emissions_tax[0][0],emissions_tax[0][1])
    #print("emissions_tax",emissions_tax[1][0],emissions_tax[1][1])
    #print("emissions_tax",emissions_tax[2][0],emissions_tax[2][1])
    print("emissions_tax shape", emissions_tax.shape)
    emissions_init = [emissions_tax[i][0][0] for i in range(len(scenarios))]
    print("emissions_init",emissions_init)
    emissions_first = [emissions_tax[i][1][0] for i in range(len(scenarios))]
    print("emissions_tax fist",emissions_first)
    print("emissions_tax init",emissions_tax[0][0][0],emissions_tax[1][0][0],emissions_tax[2][0][0])
    #


    #quit()
    #print("base_params", base_params)

    seed_reps = base_params["seed_reps"]
    
    scenario_emissions_no_tax(fileName, emissions_no_tax, scenarios,seed_reps)
    plot_scatter_end_points_emissions_scatter(fileName, emissions_tax, scenarios ,property_values_list)
    plot_means_end_points_emissions(fileName, emissions_tax, scenarios ,property_values_list)
    plot_emissions_ratio_scatter(fileName,emissions_no_tax, emissions_tax, scenarios ,property_values_list)
    plot_emissions_ratio_line(fileName,emissions_no_tax, emissions_tax, scenarios ,property_values_list)

    #print(emissions_tax[3],emissions_tax[4])
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/tax_sweep_10_56_04__08_11_2023",
    )