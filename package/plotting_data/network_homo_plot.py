"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object, calc_bounds
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.cm import get_cmap

def calculate_price_elasticity(price, emissions):
    # Calculate the percentage change in quantity demanded (emissions)
    percent_change_quantity = np.diff(emissions) / emissions[:-1]

    # Calculate the percentage change in price
    percent_change_price = np.diff(price) / price[:-1]

    # Calculate price elasticity
    price_elasticity = percent_change_quantity / percent_change_price

    return price_elasticity

def calc_price_elasticities_2D(emissions_trans, price):
    # Calculate percentage changes for each row
    percentage_change_emissions = (emissions_trans[:, 1:] - emissions_trans[:, :-1]) / emissions_trans[:, :-1] * 100

    percentage_change_price = (price[1:] - price[:-1]) / price[:-1] * 100
    # Calculate price elasticity of emissions
    price_elasticity = percentage_change_emissions / percentage_change_price

    return price_elasticity

def plot_BA_SBM_3(
    fileName: str, Data_arr_SW, Data_arr_BA, property_title, property_save, property_vals, labels_BA, Data_arr_SBM, labels_SBM, seed_reps, colors_scenarios
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11, 5), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        data_SW = Data_arr_SW[j].T
        data_BA = Data_arr_BA[j].T
        data_SBM = Data_arr_SBM[j].T
        for i in range(seed_reps):#loop through seeds
            axes[0].plot(property_vals, data_SW[i], alpha = 0.1, color = colors_scenarios[j])
            axes[1].plot(property_vals, data_SBM[i], alpha = 0.1, color = colors_scenarios[j])
            axes[2].plot(property_vals, data_BA[i], alpha = 0.1,color = colors_scenarios[j])


        mu_emissions_BA, _, _ = calc_bounds(data_BA.T, 0.95)
        mu_emissions_SBM, _, _ = calc_bounds(data_SBM.T, 0.95)
        mu_emissions_SW, _, _ = calc_bounds(data_SW.T, 0.95)

        axes[0].plot(property_vals, mu_emissions_SW, label=labels_SBM[j], color = colors_scenarios[j], alpha = 1)
        axes[1].plot(property_vals, mu_emissions_SBM, label=labels_SBM[j], color = colors_scenarios[j], alpha = 1)
        axes[2].plot(property_vals, mu_emissions_BA, label= labels_BA[j], color = colors_scenarios[j], alpha = 1)
        
        #axes[0].grid()
        #axes[1].grid()
        #axes[2].grid()
        axes[0].legend( fontsize = "8")
        axes[1].legend( fontsize = "8")
        axes[2].legend(loc= "upper right", fontsize = "8" )
        axes[0].set_title("Watt-Strogatz Small-World", fontsize = "12" )
        axes[1].set_title("Stochastic Block Model", fontsize = "12" )
        axes[2].set_title("Barabasi-Albert Scale-Free", fontsize = "12" )
    
    fig.supxlabel(property_title, fontsize = "12" )
    fig.supylabel(r"Cumulative emissions, E", fontsize = "12" )

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_BA_SBM_seeds_3" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")
    fig.savefig(f + ".eps", dpi=600, format="eps")

def plot_BA_SBM_3_alt(
    fileName: str, Data_arr_SW, Data_arr_BA, property_title, property_save, property_vals, labels_BA, Data_arr_SBM, labels_SBM, seed_reps, colors_scenarios
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11, 5), constrained_layout=True, sharey=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        data_SW = Data_arr_SW[j].T
        data_BA = Data_arr_BA[j].T
        data_SBM = Data_arr_SBM[j].T
        for i in range(seed_reps):#loop through seeds
            axes[0].plot(property_vals, data_SW[i], alpha = 0.1, color = colors_scenarios[j])
            axes[1].plot(property_vals, data_SBM[i], alpha = 0.1, color = colors_scenarios[j])
            axes[2].plot(property_vals, data_BA[i], alpha = 0.1,color = colors_scenarios[j])


        mu_emissions_BA, _, _ = calc_bounds(data_BA.T, 0.95)
        mu_emissions_SBM, _, _ = calc_bounds(data_SBM.T, 0.95)
        mu_emissions_SW, _, _ = calc_bounds(data_SW.T, 0.95)

        axes[0].plot(property_vals, mu_emissions_SW, label=labels_SBM[j], color = colors_scenarios[j], alpha = 1)
        axes[1].plot(property_vals, mu_emissions_SBM, label=labels_SBM[j], color = colors_scenarios[j], alpha = 1)
        axes[2].plot(property_vals, mu_emissions_BA, label= labels_BA[j], color = colors_scenarios[j], alpha = 1)
        
        #axes[0].grid()
        #axes[1].grid()
        #axes[2].grid()
        axes[0].legend( fontsize = "8")
        axes[1].legend( fontsize = "8")
        axes[2].legend(loc= "upper right", fontsize = "8" )
        axes[0].set_title("Watt-Strogatz Small-World", fontsize = "12" )
        axes[1].set_title("Stochastic Block Model", fontsize = "12" )
        axes[2].set_title("Barabasi-Albert Scale-Free", fontsize = "12" )
    
    fig.supxlabel(property_title, fontsize = "12" )
    fig.supylabel(r"Cumulative emissions, E", fontsize = "12" )

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_BA_SBM_seeds_3_alt" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")
    fig.savefig(f + ".eps", dpi=600, format="eps")

def plot_price_elasticies_BA_SBM_seeds_3(
    fileName: str, Data_arr_SW, Data_arr_BA, property_title, property_save, property_vals, labels_BA, Data_arr_SBM, labels_SBM, seed_reps, colors_scenarios
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        #calculate_price_elasticity
        data_SW =  (Data_arr_SW[j]).T
        data_BA =  (Data_arr_BA[j]).T
        data_SBM = (Data_arr_SBM[j]).T

        stochastic_array_price_elasticities_SW = np.asarray([calculate_price_elasticity(property_vals,x) for x in data_SW])
        stochastic_array_price_elasticities_BA = np.asarray([calculate_price_elasticity(property_vals,x) for x in data_BA])
        stochastic_array_price_elasticities_SBM = np.asarray([calculate_price_elasticity(property_vals,x) for x in  data_SBM])#calc_price_elasticities_2D((Data_arr_SBM[j]).T, property_vals_SBM)

        mean_SW = stochastic_array_price_elasticities_SW.mean(axis=0)
        mean_BA = stochastic_array_price_elasticities_BA.mean(axis=0)
        mean_SBM = stochastic_array_price_elasticities_SBM.mean(axis=0)

        axes[0].plot(property_vals[1:], mean_SW, label=labels_SBM[j], color = colors_scenarios[j], alpha = 1)
        axes[1].plot(property_vals[1:], mean_SBM, label=labels_SBM[j], color = colors_scenarios[j], alpha = 1)
        axes[2].plot(property_vals[1:], mean_BA, label= labels_BA[j], color = colors_scenarios[j], alpha = 1)

        for i in range(seed_reps):
            axes[0].plot(property_vals[1:], stochastic_array_price_elasticities_SW[i], color = colors_scenarios[j], alpha = 0.1)
            axes[1].plot(property_vals[1:], stochastic_array_price_elasticities_SBM[i], color = colors_scenarios[j], alpha = 0.1)
            axes[2].plot(property_vals[1:], stochastic_array_price_elasticities_BA[i], color = colors_scenarios[j], alpha = 0.1)

        #axes[0].grid()
        #axes[1].grid()
        #axes[2].grid()
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[0].set_title("Watt-Strogatz Small-World")
        axes[1].set_title("Stochastic Block Model")
        axes[2].set_title("Barabasi-Albert Scale-Free")
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_price_elasticies_BA_SBM_seeds_3" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")

def main(
    fileName
    ) -> None: 

    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    ############################
    #BA
    base_params = load_object(fileName + "/Data", "base_params")
    var_params = load_object(fileName + "/Data" , "var_params")
    homphily_states = load_object(fileName + "/Data" , "homphily_states")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    property_varied = var_params["property_varied"]

    emissions_array_BA = load_object(fileName + "/Data", "emissions_array_BA")
    
 
    
    emissions_array_SBM = load_object(fileName + "/Data", "emissions_array_SBM")


    emissions_array_SW = load_object(fileName + "/Data", "emissions_array_SW")
    
    #labels_BA = [r"No homophily, $h = 0$", r"Low-carbon hegemony", r"High-carbon hegemony"]
    #labels_SBM = [r"No homophily, $h = 0$", r"Low homophily, $h = %s$" % (homphily_states[1]), r"High homophily, %s$" % (homphily_states[2])]
    #labels_SBM = [r"No homophily, $h = 0$", r"Low homophily, $h = %s$" % (homphily_states[1]), r"High homophily, $h = %s$" % (homphily_states[2])]

    labels_BA = [r"No homophily", r"Low-carbon hegemony", r"High-carbon hegemony"]
    labels_SBM = [r"No homophily", r"Low homophily", r"High homophily"]
    labels_SBM = [r"No homophily", r"Low homophily", r"High homophily"]

    seed_reps = base_params["seed_reps"]

    #plot_BA_SBM_3(fileName, emissions_array_SW, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM, labels_SBM, seed_reps,colors_scenarios)
    plot_BA_SBM_3_alt(fileName, emissions_array_SW, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM, labels_SBM, seed_reps,colors_scenarios)
    #plot_price_elasticies_BA_SBM_seeds_3(fileName, emissions_array_SW, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM,  labels_SBM, seed_reps,colors_scenarios)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/networks_homo_tau_13_08_51__25_08_2024",
    )
