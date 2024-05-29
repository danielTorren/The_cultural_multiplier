
# imports
import matplotlib.pyplot as plt
from pyrsistent import v
from package.resources.utility import load_object, calc_bounds
from matplotlib.cm import  get_cmap


def plot_E_nu(
    fileName,emissions, property_vals, carbon_price_list, colors_scenarios
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(ncols = 1, nrows = 1,figsize=(15,8))#

    for i in range(len(carbon_price_list)):
        #color = next(colors)#set color for whole scenario?
        Data = emissions[i]
        #print("Data", Data.shape)
        mu_emissions, __, __ = calc_bounds(Data, 0.95)

        ax.plot(property_vals, mu_emissions, c = colors_scenarios[i], label = "Sector substitutability, $\\nu$ = %s" % (carbon_price_list[i]))

        data_trans = Data.T
        for v in range(len(data_trans)):
            ax.plot(property_vals, data_trans[v], color = colors_scenarios[i], alpha = 0.1)

    ax.set_xlabel(r"Carbon price, $\tau$")
    ax.set_ylabel(r"Cumulative emissions, E")    
    ax.legend()

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/emissions_tau"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_E_nu_mean(
    fileName,emissions, property_vals, carbon_price_list, colors_scenarios
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(ncols = 1, nrows = 1,figsize=(15,8))#

    for i in range(len(carbon_price_list)):
        #color = next(colors)#set color for whole scenario?
        Data = emissions[i]
        #print("Data", Data.shape)
        mu_emissions, __, __ = calc_bounds(Data, 0.95)

        ax.plot(property_vals, mu_emissions, c = colors_scenarios[i], label = "Sector substitutability, $\\nu$ = %s" % (carbon_price_list[i]))

        #data_trans = Data.T
        #for v in range(len(data_trans)):
        #    ax.plot(property_vals, data_trans[v], color = colors_scenarios[i], alpha = 0.1)

    ax.set_xlabel(r"Carbon price, $\tau$")
    ax.set_ylabel(r"Cumulative emissions, E")    
    ax.legend()

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/emissions_tau_mean"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps")   

def plot_E_nu_seed(
    fileName,emissions, property_vals, carbon_price_list, colors_scenarios
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(ncols = 1, nrows = 1,figsize=(15,8))#

    for i in range(len(carbon_price_list)):
        #color = next(colors)#set color for whole scenario?
        Data = emissions[i]
        #print("Data", Data.shape)
        #mu_emissions, __, __ = calc_bounds(Data, 0.95)

        #ax.plot(property_vals, mu_emissions, c = colors_scenarios[i], label = "Sector substitutability, $\\nu$ = %s" % (carbon_price_list[i]))

        data_trans = Data.T
        
        ax.plot(property_vals, data_trans[1], color = colors_scenarios[i], label = "Sector substitutability, $\\nu$ = %s" % (carbon_price_list[i]))

    ax.set_xlabel(r"Carbon price, $\tau$")
    ax.set_ylabel(r"Cumulative emissions, E") 
    ax.set_title("Seed = 1")   
    ax.legend()

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/emissions_tau_seed"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps") 


def main(
    fileName
    ) -> None: 
    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list
    ############################
    base_params = load_object(fileName + "/Data", "base_params")
    var_params = load_object(fileName + "/Data" , "var_params")
    #print("base_params", base_params)
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    property_varied = var_params["property_varied"]
    nu_list = load_object(fileName + "/Data", "nu_list")

    emissions_array = load_object(fileName + "/Data", "emissions_array")

    #print("emissions_array", emissions_array.shape)

    plot_E_nu(fileName, emissions_array,property_values_list,nu_list, colors_scenarios)
    plot_E_nu_mean(fileName, emissions_array,property_values_list,nu_list, colors_scenarios)
    plot_E_nu_seed(fileName, emissions_array,property_values_list,nu_list, colors_scenarios)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/fixed_preferences_tau_vary_12_37_58__29_05_2024",
    )
