"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
from cProfile import label
import matplotlib.pyplot as plt
from package.resources.utility import load_object, calc_bounds
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from package.generating_data.static_preferences_emissions_gen import calculate_emissions
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

def plot_price_elasticies_BA_SBM_seeds_2_3(
    fileName: str, emissions_array_BA_static, emissions_array_SBM_static, Data_arr_BA, property_title, property_save, property_vals, labels_BA, Data_arr_SBM, labels_SBM, seed_reps
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        #calculate_price_elasticity
        data_BA =  (Data_arr_BA[j]).T
        data_SBM = (Data_arr_SBM[j]).T

        stochastic_array_price_elasticities_BA = np.asarray([calculate_price_elasticity(property_vals,x) for x in data_BA])
        stochastic_array_price_elasticities_SBM = np.asarray([calculate_price_elasticity(property_vals,x) for x in  data_SBM])#calc_price_elasticities_2D((Data_arr_SBM[j]).T, property_vals_SBM)

        for i in range(seed_reps):
            axes[0][j].plot(property_vals[1:], stochastic_array_price_elasticities_BA[i])
            axes[1][j].plot(property_vals[1:], stochastic_array_price_elasticities_SBM[i], linestyle="dashed")

        emissions_array_BA_static
        static_price_elasticities_BA = calculate_price_elasticity(property_vals,emissions_array_BA_static)
        static_price_elasticities_SBM = calculate_price_elasticity(property_vals,emissions_array_SBM_static)

        axes[0][j].plot(property_vals[1:], static_price_elasticities_BA, label = "Static preferences", color = "black")
        axes[1][j].plot(property_vals[1:], static_price_elasticities_SBM, label = "Static preferences", color = "black")

        axes[0][j].set_title(labels_BA[j])
        axes[1][j].set_title(labels_SBM[j])
        axes[0][j].grid()
        axes[1][j].grid()
        axes[0][j].legend()
        axes[1][j].legend()
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Price elasticity of emissions, $\epsilon$")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_price_elasticies_BA_SBM_seeds_2_3" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")


def plot_end_points_emissions_multi_BA_SBM_2_3(
    fileName: str, emissions_array_BA_static, emissions_array_SBM_static, Data_arr_BA, property_title, property_save, property_vals, labels_BA, Data_arr_SBM, labels_SBM, seed_reps
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=2, ncols=Data_arr_BA.shape[0], figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        data_BA = Data_arr_BA[j].T
        data_SBM = Data_arr_SBM[j].T
        for i in range(seed_reps):#loop through seeds

            #for i, ax in enumerate(axes.flat):
            axes[0][j].plot(property_vals, data_BA[i])
            axes[1][j].plot(property_vals, data_SBM[i], linestyle="dashed")

        axes[0][j].plot(property_vals, emissions_array_BA_static, label = "Static preferences", color = "black")
        axes[1][j].plot(property_vals, emissions_array_SBM_static, label = "Static preferences", color = "black")

        axes[0][j].set_title(labels_BA[j])
        axes[1][j].set_title(labels_SBM[j])
        axes[0][j].grid()
        axes[1][j].grid()
        axes[0][j].legend()
        axes[1][j].legend()
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Cumulative emissions, E")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_BA_SBM_seeds_2_3" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")

########################################################################################################################

def tax_reduction(tau_list,y_line1,tau_list_full, y_line2):

    #print("inside", tau_list,y_line1,y_line2)
    # Create interpolation functions for both lines
    interp_line1 = interp1d(y_line1, tau_list, kind='linear')
    interp_line2 = interp1d(y_line2, tau_list_full, kind='linear')
    print("interp_line1", interp_line1)
    print("interp_line2", interp_line2)

    #print(tau_list,y_line1,y_line2)
    # Define the range of y values you want to consider
    y_min = max([min(y_line1)]+[min(y_line2)])
    y_max = min([max(y_line1)]+[max(y_line2)])

    #print("y_min max", y_min,y_max)
    y_values = np.linspace(y_min, y_max, 100)
    #print("y_values", y_values)
    # Calculate the x values for each y value using interpolation
    #GETS TO HERE
    x_values_line1 = interp_line1(y_values)
    x_values_line2 = interp_line2(y_values)

    # Calculate the ratio of x values for each y value
    x_ratio = x_values_line1 / x_values_line2
    #print("x_ratio", x_ratio)
    x_reduction = 1 - x_ratio

    return y_values, x_reduction

def calc_M(tau_social,tau_static):

    return 1-tau_social/tau_static

def calc_M_vector(tau_social_vec ,tau_static_vec, emissions_social, emissions_static_full):


    a = emissions_social
    b = np.asarray(emissions_static_full).flatten()
    c = tau_static_vec.flatten()

    #print(a.shape, b.shape, c.shape)
    #quit()
    predicted_tau_static_at_social_emissions = np.interp(a,b,c)

    M_vals = calc_M(tau_social_vec,predicted_tau_static_at_social_emissions)

    return M_vals

def plot_reduc_2_3(
        fileName: str,  emissions_array_static_full, total_tau_range_static, Data_arr_BA, property_title, property_save, property_vals, labels_BA, Data_arr_SBM, labels_SBM, seed_reps
    ):

    fig, axes = plt.subplots(nrows=2, ncols=3,constrained_layout=True, figsize=(14, 7))
        
    for j, Data_list in enumerate(Data_arr_BA):
        data_BA = Data_arr_BA[j].T
        data_SBM = Data_arr_SBM[j].T
        for i in range(seed_reps):#loop through seeds
            #print(property_vals.shape,data_BA[i].shape,emissions_array_BA_static.shape)
            #if j == 1 and i >= 0: 
            #    pass
            #else:
                #print(j,i)
            M_vals_BA = calc_M_vector(property_vals ,total_tau_range_static, data_BA[i], emissions_array_static_full)
            #quit()
            M_vals_SBM = calc_M_vector(property_vals ,total_tau_range_static, data_SBM[i], emissions_array_static_full)
            #for i, ax in enumerate(axes.flat):
            axes[0][j].plot(property_vals, M_vals_BA)
            axes[1][j].plot(property_vals, M_vals_SBM, linestyle="dashed")

        axes[0][j].set_title(labels_BA[j])
        axes[1][j].set_title(labels_SBM[j])
        axes[0][j].grid()
        axes[1][j].grid()

    fig.supxlabel(property_title)
    fig.supylabel(r"Carbon price reduction, M")

    plotName = fileName + "/Plots"
    f = plotName + "/tax_reduct_SBM_BA_%s" % (property_save)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 


def plot_BA_SBM_2_3(
    fileName: str, Data_arr_BA, property_title, property_save, property_vals, labels_BA, Data_arr_SBM, labels_SBM, seed_reps
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=2, ncols=Data_arr_BA.shape[0], figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        data_BA = Data_arr_BA[j].T
        data_SBM = Data_arr_SBM[j].T
        for i in range(seed_reps):#loop through seeds

            #for i, ax in enumerate(axes.flat):
            axes[0][j].plot(property_vals, data_BA[i])
            axes[1][j].plot(property_vals, data_SBM[i], linestyle="dashed")

        axes[0][j].set_title(labels_BA[j])
        axes[1][j].set_title(labels_SBM[j])
        axes[0][j].grid()
        axes[1][j].grid()
        axes[0][j].legend()
        axes[1][j].legend()
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Cumulative emissions, E")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_BA_SBM_seeds_2_3" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")
    fig.savefig(f + ".eps", dpi=600, format="eps")

def plot_BA_SBM_2(
    fileName: str, Data_arr_BA, property_title, property_save, property_vals, labels_BA, Data_arr_SBM, labels_SBM, seed_reps, colors_scenarios
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), constrained_layout=True,sharey=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        data_BA = Data_arr_BA[j].T
        data_SBM = Data_arr_SBM[j].T
        for i in range(seed_reps):#loop through seeds
            axes[0].plot(property_vals, data_BA[i], alpha = 0.2,color = colors_scenarios[j])
            axes[1].plot(property_vals, data_SBM[i], alpha = 0.2, color = colors_scenarios[j])


        mu_emissions_BA, _, _ = calc_bounds(data_BA.T, 0.95)
        mu_emissions_SBM, _, _ = calc_bounds(data_SBM.T, 0.95)

        axes[0].plot(property_vals, mu_emissions_BA, label= labels_BA[j], color = colors_scenarios[j], alpha = 1)
        axes[1].plot(property_vals, mu_emissions_SBM, label=labels_SBM[j], color = colors_scenarios[j], alpha = 1)

        axes[0].grid()
        axes[1].grid()
        axes[0].legend()
        axes[1].legend()
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Cumulative emissions, E")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_BA_SBM_seeds_2" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")
    fig.savefig(f + ".eps", dpi=600, format="eps")




def main(
    fileName
    ) -> None: 

    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list

    ############################
    #BA
    base_params_BA = load_object(fileName + "/Data", "base_params_BA")
    var_params = load_object(fileName + "/Data" , "var_params")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    property_varied = var_params["property_varied"]

    emissions_array_BA = load_object(fileName + "/Data", "emissions_array_BA")
    labels_BA = [r"BA, No homophily, $h = 0$", r"BA, Low-carbon hegemony, $h = 1$", r"BA, High-carbon hegemony, $h = 1$"]
    
    base_params_SBM = load_object(fileName + "/Data", "base_params_SBM")

    emissions_array_SBM = load_object(fileName + "/Data", "emissions_array_SBM")
    labels_SBM = [r"SBM, No homophily, $h = 0$", r"SBM, Low homophily, $h = 0.5$", r"SBM, High homophily, $h = 1$"]


    seed_reps = base_params_BA["seed_reps"]

    #reference_run = load_object(fileName + "/Data", "reference_run")
    #emissions_array_static = load_object(fileName + "/Data", "emissions_array_static")
    #emissions_array_static_full = load_object(fileName + "/Data", "emissions_array_static_full")
    #total_tau_range_static = load_object(fileName + "/Data", "total_tau_range_static")

    #plot_end_points_emissions_multi_BA_SBM_2_3(fileName, emissions_array_static, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM, labels_SBM, seed_reps)
    #plot_price_elasticies_BA_SBM_seeds_2_3(fileName, emissions_array_static, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM,  labels_SBM, seed_reps)
    #plot_reduc_2_3(fileName, emissions_array_static_full, total_tau_range_static, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM,  labels_SBM, seed_reps)
    
    #plot_BA_SBM_2_3(fileName, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM, labels_SBM, seed_reps)
    plot_BA_SBM_2(fileName, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM, labels_SBM, seed_reps,colors_scenarios)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/BA_SBM_tau_vary_22_39_24__06_04_2024",
    )
