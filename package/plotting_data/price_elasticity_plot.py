"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
from cProfile import label
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from package.resources.utility import load_object
from package.resources.plot import (
    plot_end_points_emissions,
    plot_end_points_emissions_scatter,
    plot_end_points_emissions_lines,
)
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable
from scipy.signal import savgol_filter, butter, filtfilt


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


def plot_end_points_emissions_multi_BA_SBM(
    fileName: str, Data_arr_BA, property_title, property_save_BA, property_vals_BA, labels_BA, Data_arr_SBM, property_save_SBM, property_vals_SBM, labels_SBM
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6),constrained_layout = True )

    #BA
    for i, Data_list in enumerate(Data_arr_BA):
        mu_emissions =  Data_list.mean(axis=1)
        min_emissions =  Data_list.min(axis=1)
        max_emissions=  Data_list.max(axis=1)

        ax.plot(property_vals_BA, mu_emissions, label = labels_BA[i])
        ax.fill_between(property_vals_BA, min_emissions, max_emissions, alpha=0.5)

    #SBM
    for i, Data_list in enumerate(Data_arr_SBM):
        mu_emissions =  Data_list.mean(axis=1)
        min_emissions =  Data_list.min(axis=1)
        max_emissions=  Data_list.max(axis=1)

        ax.plot(property_vals_SBM, mu_emissions, label = labels_SBM[i], linestyle="dashed")
        ax.fill_between(property_vals_SBM, min_emissions, max_emissions, alpha=0.5)

    ax.legend()
    ax.set_xlabel(property_title)
    ax.set_ylabel(r"Cumulative carbon emissions, E")

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/multi_" + property_save_BA + "_"+ property_save_SBM+ "_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_price_elasticies_BA_SBM(
    fileName: str, Data_arr_BA, property_title, property_save_BA, property_vals_BA, labels_BA, Data_arr_SBM, property_save_SBM, property_vals_SBM, labels_SBM
):

    fig, ax = plt.subplots(figsize=(10,6),constrained_layout = True )

    #BA
    """BADLY CODED SO I TAKE A TRANSPOSE TWICE; FIX THIS AT SOME POINT"""
    for i, Data_list in enumerate(Data_arr_BA):
        stochastic_array_price_elasticties = (calc_price_elasticities_2D(Data_list.T, property_vals_BA)).T
        mu_emissions =  stochastic_array_price_elasticties.mean(axis=1)
        #min_emissions =  stochastic_array_price_elasticties.min(axis=1)
        #max_emissions =  stochastic_array_price_elasticties.max(axis=1)

        ax.plot(property_vals_BA[1:], mu_emissions, label = labels_BA[i])
        #ax.fill_between(property_vals_BA[1:], min_emissions, max_emissions, alpha=0.5)

    #SBM
    for i, Data_list in enumerate(Data_arr_SBM):
        stochastic_array_price_elasticties = (calc_price_elasticities_2D(Data_list.T, property_vals_SBM)).T

        mu_emissions =  stochastic_array_price_elasticties.mean(axis=1)
        #min_emissions =  stochastic_array_price_elasticties.min(axis=1)
        #max_emissions =  stochastic_array_price_elasticties.max(axis=1)

        ax.plot(property_vals_SBM[1:], mu_emissions, label = labels_SBM[i], linestyle="dashed")
        #ax.fill_between(property_vals_SBM[1:], min_emissions, max_emissions, alpha=0.5)

    ax.legend()
    ax.set_xlabel(property_title)
    ax.set_ylabel(r"Price elasticity of emissions, $\epsilon$")

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/plot_price_elasticies_BA_SBM_" + property_save_BA + "_"+ property_save_SBM
    fig.savefig(f+ ".png", dpi=600, format="png")  


def plot_price_elasticies_BA_SBM_seeds(
    fileName: str, Data_arr_BA, property_title, property_save_BA, property_vals_BA, labels_BA, Data_arr_SBM, property_save_SBM, property_vals_SBM, labels_SBM, seed_reps
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 6), constrained_layout=True, sharex=True, sharey=True)

    for j, Data_list in enumerate(Data_arr_BA):

        stochastic_array_price_elasticities_BA = calc_price_elasticities_2D((Data_arr_BA[j]).T, property_vals_BA)
        stochastic_array_price_elasticities_SBM = calc_price_elasticities_2D((Data_arr_SBM[j]).T, property_vals_SBM)
        for i, ax in enumerate(axes.flat):
            ax.plot(property_vals_BA[1:], stochastic_array_price_elasticities_BA[i], label=labels_BA[j])
            ax.plot(property_vals_SBM[1:], stochastic_array_price_elasticities_SBM[i], label=labels_SBM[j], linestyle="dashed")

            ax.grid()
    
    axes[-1].set_xlabel(property_title)
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    #CANT SEE IT FOR SOME REASON!!!!
    # Add a common legend to the right of all subplots
    lines, labels = axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))

    plotName = fileName + "/Plots"
    f = plotName + "/plot_price_elasticies_BA_SBM_seeds_" + property_save_BA + "_" + property_save_SBM
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_price_elasticies_BA_SBM_seeds_2_3(
    fileName: str, Data_arr_BA, property_title, property_save_BA, property_vals_BA, labels_BA, Data_arr_SBM, property_save_SBM, property_vals_SBM, labels_SBM, seed_reps
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        #calculate_price_elasticity
        data_BA =  (Data_arr_BA[j]).T
        data_SBM = (Data_arr_SBM[j]).T

        stochastic_array_price_elasticities_BA = np.asarray([calculate_price_elasticity(property_vals_BA,x) for x in data_BA])
        stochastic_array_price_elasticities_SBM = np.asarray([calculate_price_elasticity(property_vals_SBM,x) for x in  data_SBM])#calc_price_elasticities_2D((Data_arr_SBM[j]).T, property_vals_SBM)

        for i in range(seed_reps):
            axes[0][j].plot(property_vals_BA[1:], stochastic_array_price_elasticities_BA[i])
            axes[1][j].plot(property_vals_SBM[1:], stochastic_array_price_elasticities_SBM[i], linestyle="dashed")

        axes[0][j].set_title(labels_BA[j])
        axes[1][j].set_title(labels_SBM[j])
        axes[0][j].grid()
        axes[1][j].grid()
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    #CANT SEE IT FOR SOME REASON!!!!
    # Add a common legend to the right of all subplots
    #lines, labels = axes[-1].get_legend_handles_labels()
    #fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))

    plotName = fileName + "/Plots"
    f = plotName + "/plot_price_elasticies_BA_SBM_seeds_2_3" + property_save_BA + "_" + property_save_SBM
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_av_price_elasticies_BA_SBM_seeds_2_3(
    fileName: str, Data_arr_BA, property_title, property_save_BA, property_vals_BA, labels_BA, Data_arr_SBM, property_save_SBM, property_vals_SBM, labels_SBM, seed_reps,av_num
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):

        data_BA = np.asarray([moving_average(x,av_num) for x in (Data_arr_BA[j]).T])

        prop_vals_BA_init = property_vals_BA.shape[0] - data_BA.shape[1]

        stochastic_array_price_elasticities_BA = calc_price_elasticities_2D(data_BA , property_vals_BA[prop_vals_BA_init:])
        stochastic_array_price_elasticities_SBM = calc_price_elasticities_2D((Data_arr_SBM[j]).T, property_vals_SBM)

        #quit()
        for i in range(seed_reps):
        #for i, ax in enumerate(axes.flat):
            
            #print("prop_vals_BA_init",prop_vals_BA_init)
            axes[0][j].plot(property_vals_BA[1+prop_vals_BA_init:], stochastic_array_price_elasticities_BA[i])
            axes[1][j].plot(property_vals_SBM[1:], stochastic_array_price_elasticities_SBM[i], linestyle="dashed")

        axes[0][j].set_title(labels_BA[j])
        axes[1][j].set_title(labels_SBM[j])
        axes[0][j].grid()
        axes[1][j].grid()
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_av_price_elasticies_BA_SBM_seeds_2_3" + property_save_BA + "_" + property_save_SBM
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_end_points_emissions_multi_BA_SBM_2_3(
    fileName: str, Data_arr_BA, property_title, property_save_BA, property_vals_BA, labels_BA, Data_arr_SBM, property_save_SBM, property_vals_SBM, labels_SBM, seed_reps
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=2, ncols=Data_arr_BA.shape[0], figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        data_BA = Data_arr_BA[j].T
        data_SBM = Data_arr_SBM[j].T
        for i in range(seed_reps):#loop through seeds

            #for i, ax in enumerate(axes.flat):
            axes[0][j].plot(property_vals_BA, data_BA[i])
            axes[1][j].plot(property_vals_SBM, data_SBM[i], linestyle="dashed")

            axes[0][j].set_title(labels_BA[j])
            axes[1][j].set_title(labels_SBM[j])
            axes[0][j].grid()
            axes[1][j].grid()
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_BA_SBM_seeds_2_3" + property_save_BA + "_" + property_save_SBM
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_scatter_end_emissions_multi_BA_SBM_2_3(
    fileName: str, Data_arr_BA, property_title, property_save_BA, property_vals_BA, labels_BA, Data_arr_SBM, property_save_SBM, property_vals_SBM, labels_SBM, seed_reps
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=2, ncols=Data_arr_BA.shape[0], figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):

        data_BA = Data_arr_BA[j].T
        data_SBM = Data_arr_SBM[j].T
        for i in range(seed_reps):#loop through seeds

            #for i, ax in enumerate(axes.flat):
            axes[0][j].scatter(property_vals_BA, data_BA[i], marker='.')
            axes[1][j].scatter(property_vals_SBM, data_SBM[i],  marker='.')

            axes[0][j].set_title(labels_BA[j])
            axes[1][j].set_title(labels_SBM[j])
            axes[0][j].grid()
            axes[1][j].grid()
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    #CANT SEE IT FOR SOME REASON!!!!
    # Add a common legend to the right of all subplots
    #lines, labels = axes[-1].get_legend_handles_labels()
    #fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))

    plotName = fileName + "/Plots"
    f = plotName + "/plot_scatter_emissions_BA_SBM_seeds_2_3" + property_save_BA + "_" + property_save_SBM
    fig.savefig(f + ".png", dpi=600, format="png")

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
#########################################################################################################################


def plot_savgol_price_elasticies_BA_SBM_seeds_2_3(
    fileName: str, Data_arr_BA, property_title, property_save_BA, property_vals_BA, labels_BA, Data_arr_SBM, property_save_SBM, property_vals_SBM, labels_SBM, seed_reps, window_size, poly
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        
        smoothed_data_sg = np.asarray([savgol_filter(x, window_size, poly) for x in (Data_arr_BA[j]).T])   # Adjust the polynomial order if needed
        #print(smoothed_data_sg.shape)
        #quit()

        #prop_vals_BA_init = property_vals_BA.shape[0] - data_BA.shape[1]

        stochastic_array_price_elasticities_BA = calc_price_elasticities_2D(smoothed_data_sg , property_vals_SBM)
        stochastic_array_price_elasticities_SBM = calc_price_elasticities_2D((Data_arr_SBM[j]).T, property_vals_SBM)

        #quit()
        for i in range(seed_reps):
        #for i, ax in enumerate(axes.flat):
            
            #print("prop_vals_BA_init",prop_vals_BA_init)
            axes[0][j].plot(property_vals_BA[1:], stochastic_array_price_elasticities_BA[i])
            axes[1][j].plot(property_vals_SBM[1:], stochastic_array_price_elasticities_SBM[i], linestyle="dashed")

        axes[0][j].set_title(labels_BA[j])
        axes[1][j].set_title(labels_SBM[j])
        axes[0][j].grid()
        axes[1][j].grid()
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_savgol_price_elasticies_BA_SBM_seeds_2_3" + property_save_BA + "_" + property_save_SBM
    fig.savefig(f + ".png", dpi=600, format="png")

def main(
    fileName_BA,
    fileName_SBM
    ) -> None: 

    ############################
    #BA
    base_params_BA = load_object(fileName_BA + "/Data", "base_params")
    #print("base_params_BA",base_params_BA)
    var_params_BA  = load_object(fileName_BA + "/Data" , "var_params")
    property_values_list_BA = load_object(fileName_BA + "/Data", "property_values_list")
    property_varied_BA = var_params_BA["property_varied"]
    emissions_array_BA = load_object(fileName_BA + "/Data", "emissions_array")
    labels_BA = [r"BA, No homophily, $h = 0$", r"BA, High-carbon hegemony, $h = 1$",r"BA, Low-carbon hegemony, $h = 1$"]
    #print("INFO BA")
    #print("base_params_BA", base_params_BA)
    #print("var_params_BA", var_params_BA)
    #quit()
    #SBM
    base_params_SBM = load_object(fileName_SBM + "/Data", "base_params")
    #print("base_params_SBM", base_params_SBM)
    var_params_SBM = load_object(fileName_SBM + "/Data", "var_params")
    property_values_list_SBM = load_object(fileName_SBM + "/Data", "property_values_list")
    property_varied_SBM = var_params_SBM["property_varied"]
    emissions_array_SBM = load_object(fileName_SBM + "/Data", "emissions_array")
    labels_SBM = [r"SBM, No homophily, $h = 0$", r"SBM, Low homophily, $h = 0.5$", r"SBM, High homophily, $h = 1$"]

    #print("INFO SBM")
    #print("base_params_SBM", base_params_SBM)
    #print("var_params_SBM", var_params_SBM)

    seed_reps = base_params_BA["seed_reps"]

    #plot_end_points_emissions_multi_BA_SBM(fileName_BA, emissions_array_BA, r"Carbon price, $\tau$", property_varied_BA, property_values_list_BA, labels_BA, emissions_array_SBM, property_varied_SBM, property_values_list_SBM, labels_SBM)
    #plot_end_points_emissions_multi_BA_SBM_2_3(fileName_BA, emissions_array_BA, r"Carbon price, $\tau$", property_varied_BA, property_values_list_BA, labels_BA, emissions_array_SBM, property_varied_SBM, property_values_list_SBM, labels_SBM,seed_reps)
    #plot_scatter_end_emissions_multi_BA_SBM_2_3(fileName_BA, emissions_array_BA, r"Carbon price, $\tau$", property_varied_BA, property_values_list_BA, labels_BA, emissions_array_SBM, property_varied_SBM, property_values_list_SBM, labels_SBM,seed_reps)
    #plot_price_elasticies_BA_SBM(fileName_BA, emissions_array_BA, r"Carbon price, $\tau$", property_varied_BA, property_values_list_BA, labels_BA, emissions_array_SBM, property_varied_SBM, property_values_list_SBM, labels_SBM)
    #plot_price_elasticies_BA_SBM_seeds(fileName_BA, emissions_array_BA, r"Carbon price, $\tau$", property_varied_BA, property_values_list_BA, labels_BA, emissions_array_SBM, property_varied_SBM, property_values_list_SBM, labels_SBM,seed_reps)
    plot_price_elasticies_BA_SBM_seeds_2_3(fileName_BA, emissions_array_BA, r"Carbon price, $\tau$", property_varied_BA, property_values_list_BA, labels_BA, emissions_array_SBM, property_varied_SBM, property_values_list_SBM, labels_SBM,seed_reps)
    #av_num = 200
    #plot_av_price_elasticies_BA_SBM_seeds_2_3(fileName_BA, emissions_array_BA, r"Carbon price, $\tau$", property_varied_BA, property_values_list_BA, labels_BA, emissions_array_SBM, property_varied_SBM, property_values_list_SBM, labels_SBM,seed_reps,av_num)
    #window_size = 50
    #poly = 4
    #plot_savgol_price_elasticies_BA_SBM_seeds_2_3(fileName_BA, emissions_array_BA, r"Carbon price, $\tau$", property_varied_BA, property_values_list_BA, labels_BA, emissions_array_SBM, property_varied_SBM, property_values_list_SBM, labels_SBM,seed_reps,window_size, poly)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName_BA= "results/BA_heg_tau_vary_14_51_56__17_01_2024",
        fileName_SBM= "results/SBM_tau_vary_14_53_24__17_01_2024",
    )
