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


def plot_price_elasticies_SF_SBM_seeds_3(
    fileName: str, Data_arr_SW, Data_arr_SF, property_title, property_save, property_vals, labels_SF, Data_arr_SBM, labels_SBM, seed_reps, colors_scenarios
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_SF):
        #calculate_price_elasticity
        data_SW =  (Data_arr_SW[j]).T
        data_SF =  (Data_arr_SF[j]).T
        data_SBM = (Data_arr_SBM[j]).T

        stochastic_array_price_elasticities_SW = np.asarray([calculate_price_elasticity(property_vals,x) for x in data_SW])
        stochastic_array_price_elasticities_SF = np.asarray([calculate_price_elasticity(property_vals,x) for x in data_SF])
        stochastic_array_price_elasticities_SBM = np.asarray([calculate_price_elasticity(property_vals,x) for x in  data_SBM])#calc_price_elasticities_2D((Data_arr_SBM[j]).T, property_vals_SBM)

        mean_SW = stochastic_array_price_elasticities_SW.mean(axis=0)
        mean_SF = stochastic_array_price_elasticities_SF.mean(axis=0)
        mean_SBM = stochastic_array_price_elasticities_SBM.mean(axis=0)

        axes[0].plot(property_vals[1:], mean_SW, label=labels_SBM[j], color = colors_scenarios[j], alpha = 1)
        axes[1].plot(property_vals[1:], mean_SBM, label=labels_SBM[j], color = colors_scenarios[j], alpha = 1)
        axes[2].plot(property_vals[1:], mean_SF, label= labels_SF[j], color = colors_scenarios[j], alpha = 1)

        for i in range(seed_reps):
            axes[0].plot(property_vals[1:], stochastic_array_price_elasticities_SW[i], color = colors_scenarios[j], alpha = 0.1)
            axes[1].plot(property_vals[1:], stochastic_array_price_elasticities_SBM[i], color = colors_scenarios[j], alpha = 0.1)
            axes[2].plot(property_vals[1:], stochastic_array_price_elasticities_SF[i], color = colors_scenarios[j], alpha = 0.1)

        axes[0].grid()
        axes[1].grid()
        axes[2].grid()
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[0].set_title("Small-World")
        axes[1].set_title("Stochastic Block Model")
        axes[2].set_title("Scale-Free")
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_price_elasticies_SF_SBM_seeds_3" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")



def plot_SW_SBM(
    fileName: str, Data_arr_SW, Data_arr_SBM, property_title, property_save, property_vals, labels_SBM, seed_reps, colors_scenarios, emissions_fixed
):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), constrained_layout=True, sharey=True)

    mu_emissions, lower_bound, upper_bound = calc_bounds(emissions_fixed, 0.95)

    for i in [0,1]:
        axes[i].plot(property_vals, mu_emissions, color="black", alpha=1, linestyle= "dashed")
        #axes[i].fill_between(property_vals, lower_bound, upper_bound, color="black", alpha=0.3)

    for j, Data_list in enumerate(Data_arr_SW):
        data_SW = Data_arr_SW[j].T
        data_SBM = Data_arr_SBM[j].T

        mu_emissions_SW, lower_bound_SW, upper_bound_SW = calc_bounds(data_SW.T, 0.95)
        mu_emissions_SBM, lower_bound_SBM, upper_bound_SBM = calc_bounds(data_SBM.T, 0.95)
        
        emissions_fixed
        axes[0].plot(property_vals, mu_emissions_SW, label=labels_SBM[j], color=colors_scenarios[j], alpha=1)
        axes[1].plot(property_vals, mu_emissions_SBM, label=labels_SBM[j], color=colors_scenarios[j], alpha=1)

        axes[0].fill_between(property_vals, lower_bound_SW, upper_bound_SW, color=colors_scenarios[j], alpha=0.3)
        axes[1].fill_between(property_vals, lower_bound_SBM, upper_bound_SBM, color=colors_scenarios[j], alpha=0.3)

        #axes[0].legend(fontsize="8")
        axes[1].legend(fontsize="8")
        axes[0].set_title("Small-World", fontsize="12")
        axes[1].set_title("Stochastic Block Model", fontsize="12")
        
    axes[0].grid()
    axes[1].grid()
    fig.supxlabel(property_title, fontsize="12")
    fig.supylabel(r"Cumulative emissions, E", fontsize="12")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_SW_SBM_seeds_" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")
    fig.savefig(f + ".eps", dpi=600, format="eps")

def plot_SF(
    fileName: str, Data_arr_SF, property_title, property_save, property_vals, labels_SF, seed_reps, colors_scenarios,  emissions_fixed
):
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

    mu_emissions, lower_bound, upper_bound = calc_bounds(emissions_fixed, 0.95)
    ax.plot(property_vals, mu_emissions, color="black", alpha=1, linestyle= "dashed")
    #ax.fill_between(property_vals, lower_bound, upper_bound, color="black", alpha=0.3)

    for j, Data_list in enumerate(Data_arr_SF):
        data_SF = Data_arr_SF[j].T
        mu_emissions_SF, lower_bound_SF, upper_bound_SF = calc_bounds(data_SF.T, 0.95)
        
        ax.plot(property_vals, mu_emissions_SF, label=labels_SF[j], color=colors_scenarios[j], alpha=1)
        ax.fill_between(property_vals, lower_bound_SF, upper_bound_SF, color=colors_scenarios[j], alpha=0.3)


        ax.legend(loc="upper right", fontsize="8")
        ax.set_title("Scale-Free", fontsize="12")
    ax.grid()
    fig.supxlabel(property_title, fontsize="12")
    fig.supylabel(r"Cumulative emissions, E", fontsize="12")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_SF_seeds_" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")
    fig.savefig(f + ".eps", dpi=600, format="eps")


def main(
    fileName,
    fileName_fixed
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

    emissions_array_SF = load_object(fileName + "/Data", "emissions_array_SF")    
    #print(emissions_array_SF.shape)
    #quit()
    emissions_array_SBM = load_object(fileName + "/Data", "emissions_array_SBM")
    emissions_array_SW = load_object(fileName + "/Data", "emissions_array_SW")
    
    emissions_fixed = load_object(fileName_fixed + "/Data","emissions_SW")[0]

    #labels_SF = [r"No homophily, $h = 0$", r"Low-carbon hegemony", r"High-carbon hegemony"]
    #labels_SBM = [r"No homophily, $h = 0$", r"Low homophily, $h = %s$" % (homphily_states[1]), r"High homophily, %s$" % (homphily_states[2])]
    #labels_SBM = [r"No homophily, $h = 0$", r"Low homophily, $h = %s$" % (homphily_states[1]), r"High homophily, $h = %s$" % (homphily_states[2])]

    labels_SF = [r"No homophily", r"Low-carbon hegemony", r"High-carbon hegemony"]
    labels_SBM = [r"No homophily", r"Low homophily", r"High homophily"]
    labels_SBM = [r"No homophily", r"Low homophily", r"High homophily"]

    seed_reps = base_params["seed_reps"]
    property_title = r"Carbon tax, $\tau$"
    #plot_SF_SBM_3(fileName, emissions_array_SW, emissions_array_SF, r"Carbon tax, $\tau$", property_varied, property_values_list, labels_SF, emissions_array_SBM, labels_SBM, seed_reps,colors_scenarios)
    #plot_SF_SBM_3_alt(fileName, emissions_array_SW, emissions_array_SF, r"Carbon tax, $\tau$", property_varied, property_values_list, labels_SF, emissions_array_SBM, labels_SBM, seed_reps,colors_scenarios)
    #plot_price_elasticies_SF_SBM_seeds_3(fileName, emissions_array_SW, emissions_array_SF, r"Carbon tax, $\tau$", property_varied, property_values_list, labels_SF, emissions_array_SBM,  labels_SBM, seed_reps,colors_scenarios)
    plot_SW_SBM(fileName, emissions_array_SW, emissions_array_SBM, r"Carbon tax, $\tau$", property_varied, property_values_list, labels_SBM, seed_reps,colors_scenarios, emissions_fixed)
    plot_SF(fileName, emissions_array_SF , property_title, property_varied, property_values_list, labels_SF, seed_reps, colors_scenarios, emissions_fixed)


    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/networks_homo_tau_13_08_51__25_08_2024",
        fileName_fixed = "results/tax_sweep_networks_15_57_56__22_08_2024"
    )
