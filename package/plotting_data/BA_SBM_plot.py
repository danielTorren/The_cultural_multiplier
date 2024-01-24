"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from package.generating_data.static_preferences_emissions_gen import calculate_emissions

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

def trying_to_plot_theos_reduction(tau_list,y_line1,y_line2):

    #print("inside", tau_list,y_line1,y_line2)
    # Create interpolation functions for both lines
    interp_line1 = interp1d(y_line1, tau_list, kind='linear')
    interp_line2 = interp1d(y_line2, tau_list, kind='linear')
    #print("interp_line1", interp_line1)
    #print("interp_line2", interp_line2)

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

def plot_reduc_2_3(
        fileName: str, emissions_array_BA_static, emissions_array_SBM_static, Data_arr_BA, property_title, property_save, property_vals, labels_BA, Data_arr_SBM, labels_SBM, seed_reps
    ):

    fig, axes = plt.subplots(nrows=2, ncols=3,constrained_layout=True, figsize=(14, 7))
        
    for j, Data_list in enumerate(Data_arr_BA):
        data_BA = Data_arr_BA[j].T
        data_SBM = Data_arr_SBM[j].T
        for i in range(seed_reps):#loop through seeds
            #print(property_vals.shape,data_BA[i].shape,emissions_array_BA_static.shape)
            if j == 1 and i >= 0: 
                pass
            else:
                print(j,i)
                y_values_social_BA, x_reduction_social_BA = trying_to_plot_theos_reduction(property_vals, data_BA[i], emissions_array_BA_static)
                y_values_social_SBM, x_reduction_social_SBM = trying_to_plot_theos_reduction(property_vals, data_SBM[i], emissions_array_SBM_static)
                #for i, ax in enumerate(axes.flat):
                axes[0][j].plot(y_values_social_BA, x_reduction_social_BA)
                axes[1][j].plot(y_values_social_SBM, x_reduction_social_SBM, linestyle="dashed")

        axes[0][j].set_title(labels_BA[j])
        axes[1][j].set_title(labels_SBM[j])
        axes[0][j].grid()
        axes[1][j].grid()

    fig.supxlabel(r"Cumulative emissions, E")
    fig.supylabel(r"Carbon price reduction")

    plotName = fileName + "/Plots"
    f = plotName + "/tax_reduct_SBM_BA_%s" % (property_save)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 



##################################################################################################
#REVERSE Engineer the carbon price based on the final emissions
def objective_function(P_H, *args):
    emissions_val, t_max, B, N, M, a, P_L, A, sigma, nu = args
    E = calculate_emissions(t_max, B, N, M, a, P_L, P_H, A, sigma, nu)
    return emissions_val - E 

def optimize_PH(emissions_val, t_max, B, N, M, a, P_L, A, sigma, nu, initial_guess):

    args = (emissions_val,t_max, B, N, M, a, P_L, A, sigma, nu)

    # Set bounds for P_H values (you can adjust the bounds as needed)
    bounds = [(-np.inf, np.inf)]
    result = minimize(objective_function, initial_guess, args=args, bounds=bounds)

    if result.success:
        optimal_PH = result.x
        return optimal_PH
    else:
        raise ValueError(f"Optimization failed: {result.message}")


def calc_fitted_emissions_static_preference(reps,emissions_min, emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, initial_guess):
    
    min_P_H = optimize_PH(emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, initial_guess)
    max_P_H = optimize_PH(emissions_min, t_max, B, N, M, a, P_L, A, sigma, nu, initial_guess)

    P_H_array = np.linspace(min_P_H,max_P_H, reps)
    E_list = [calculate_emissions(t_max, B, N, M, a, P_L, P_H, A, sigma, nu) for P_H in P_H_array]
    return E_list




def main(
    fileName
    ) -> None: 

    ############################
    #BA
    base_params_BA = load_object(fileName + "/Data", "base_params_BA")
    #print("base_params_BA",base_params_BA)
    var_params = load_object(fileName + "/Data" , "var_params")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    property_varied = var_params["property_varied"]

    emissions_array_BA = load_object(fileName + "/Data", "emissions_array_BA")
    emissions_array_BA_static = load_object(fileName + "/Data", "emissions_array_BA_static")
    labels_BA = [r"BA, No homophily, $h = 0$", r"BA, Low-carbon hegemony, $h = 1$", r"BA, High-carbon hegemony, $h = 1$"]
    
    base_params_SBM = load_object(fileName + "/Data", "base_params_SBM")

    emissions_array_SBM = load_object(fileName + "/Data", "emissions_array_SBM")
    emissions_array_SBM_static = load_object(fileName + "/Data", "emissions_array_SBM_static")
    labels_SBM = [r"SBM, No homophily, $h = 0$", r"SBM, Low homophily, $h = 0.5$", r"SBM, High homophily, $h = 1$"]


    seed_reps = base_params_BA["seed_reps"]

    reference_run = load_object(fileName + "/Data", "reference_run")
    #calc the fixed emissions
    emissions_min =  np.min([np.min(emissions_array_BA),np.min(emissions_array_SBM)])
    emissions_max = np.max([np.max(emissions_array_BA),np.max(emissions_array_SBM)])
    
    #print("emissions_min", emissions_min)
    #print("emissions_max", emissions_max)

    initial_guess  = 1
    t_max, B, N, M, a, P_L, A, sigma, nu= (
        reference_run.carbon_price_duration + reference_run.burn_in_duration,
        reference_run.expenditure,
        reference_run.N,
        reference_run.M,
        reference_run.sector_preferences,
        reference_run.prices_low_carbon,
        reference_run.low_carbon_preference_matrix_init,
        np.asarray(reference_run.low_carbon_substitutability_array_list),
        reference_run.sector_substitutability
        )
    #print("vals", t_max, B, N, M, a, P_L, sigma, nu)
    #print("base_params_BA", base_params_BA["carbon_price_duration"]+base_params_BA["burn_in_duration"], base_params_BA["expenditure"], base_params_BA["N"],base_params_BA["M"])
    #quit()
    
    E_list = [calculate_emissions(t_max, B, N, M, a, P_L, (1 + tau), A, sigma, nu) for tau in property_values_list]
    #SO THIS IS JUST WRONG
    print("E_list", E_list)
    print("emissions_array_SBM_static", emissions_array_SBM_static)
    print("emissions_array_BA_static",emissions_array_BA_static)
    print(emissions_array_BA_static/np.asarray(E_list))
    quit()

    calc_fitted_emissions_static_preference(1000,emissions_min, emissions_max, t_max, B, N, M, a, P_L, A, sigma, nu, initial_guess)

    #plot_end_points_emissions_multi_BA_SBM_2_3(fileName, emissions_array_BA_static, emissions_array_SBM_static, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM, labels_SBM, seed_reps)
    #plot_price_elasticies_BA_SBM_seeds_2_3(fileName, emissions_array_BA_static, emissions_array_SBM_static, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM,  labels_SBM, seed_reps)
    #plot_reduc_2_3(fileName, emissions_array_BA_static, emissions_array_SBM_static, emissions_array_BA, r"Carbon price, $\tau$", property_varied, property_values_list, labels_BA, emissions_array_SBM,  labels_SBM, seed_reps)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/BA_SBM_tau_vary_17_37_07__24_01_2024",
    )
