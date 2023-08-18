
# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from package.resources.utility import get_cmap_colours
from scipy.interpolate import interp1d
import numpy as np

def trying_to_plot_theos_reduction(tau_list,y_line1,y_line2):

    #print("inside", tau_list,y_line1,y_line2)
    # Create interpolation functions for both lines
    interp_line1 = interp1d(y_line1, tau_list, kind='linear')
    interp_line2 = interp1d(y_line2, tau_list, kind='linear')

    #print(tau_list,y_line1,y_line2)
    # Define the range of y values you want to consider
    y_min = max([min(y_line1)]+[min(y_line2)])
    y_max = min([max(y_line1)]+[max(y_line2)])

    #print("y_min max", y_min,y_max)
    y_values = np.linspace(y_min, y_max, 100)

    # Calculate the x values for each y value using interpolation
    x_values_line1 = interp_line1(y_values)
    x_values_line2 = interp_line2(y_values)

    # Calculate the ratio of x values for each y value
    x_ratio = x_values_line1 / x_values_line2
    #print("x_ratio", x_ratio)
    x_reduction = 1 - x_ratio

    return y_values, x_reduction

def plot_multipliers(fileName,data_array_social,data_array_cultural, tau_list, phi_list,y_title,save_type):
    
    fig, axes = plt.subplots(nrows=1, ncols=2,constrained_layout=True, figsize=(14, 7), sharex=True, sharey=True)

    for i, phi in enumerate(phi_list):
        axes[0].plot(tau_list, data_array_social[i], label = "$\phi$ = %s" %(round(phi,3)))
        axes[1].plot(tau_list, data_array_cultural[i], label = "$\phi$ = %s" %(round(phi,3)))
    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlabel(r"Carbon tax, $\tau$")#col
    axes[1].set_xlabel(r"Carbon tax, $\tau$")#col
    axes[0].set_ylabel(y_title)#row
    
    axes[0].set_title("Social Multiplier")
    axes[1].set_title("Cultural Multiplier")

    plotName = fileName + "/Plots"
    f = plotName + "/sorted_diif_multiplier_varying_tax_%s" % (save_type)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

def plot_reduc(fileName,data_array_social,data_array_cultural, tau_list, phi_list, y_title,save_type):

    fig, axes = plt.subplots(nrows=1, ncols=2,constrained_layout=True, figsize=(14, 7), sharex=True, sharey=True)

    for i in range(len(phi_list) - 1):
        fixed_social = data_array_social[0]
        var_social = data_array_social[i+1]
        y_values_social, x_reduction_social= trying_to_plot_theos_reduction(tau_list,var_social,fixed_social)
        axes[0].plot(y_values_social, x_reduction_social, label = "$\phi$ = %s" %(round(phi_list[i+1],3)))
    
        fixed_cultural = data_array_cultural[0]
        var_cultural = data_array_cultural[i+1]
        y_values_cultural, x_reduction_cultural= trying_to_plot_theos_reduction(tau_list,var_cultural,fixed_cultural)
        axes[1].plot(y_values_cultural, x_reduction_cultural, label = "$\phi$ = %s" %(round(phi_list[i+1],3)))

    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlabel(y_title)#col
    axes[1].set_xlabel(y_title)#col
    axes[0].set_ylabel("Tax reduction")#row
    
    axes[0].set_title("Social Multiplier")
    axes[1].set_title("Cultural Multiplier")


    plotName = fileName + "/Plots"
    f = plotName + "/tax_reduct_sorted_%s" % (save_type)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

def main(
    fileName = "results/deriving_multipliers_14_49_23__01_08_2023",
    ) -> None: 

    ############################

    #data_holder_stock_social_multiplier = load_object(fileName + "/Data", "data_holder_stock_social_multiplier")
    data_holder_flow_social_multiplier = load_object(fileName + "/Data", "data_holder_flow_social_multiplier")
    #data_holder_stock_cultural_multiplier = load_object(fileName + "/Data", "data_holder_stock_cultural_multiplier")
    data_holder_flow_cultural_multiplier = load_object(fileName + "/Data", "data_holder_flow_cultural_multiplier")

    #property_varied = load_object(fileName + "/Data", "property_varied")
    #property_varied_title = load_object(fileName + "/Data", "property_varied_title")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    #base_params = load_object(fileName + "/Data", "base_params")
    phi_list = load_object(fileName + "/Data", "phi_list")

    #print("base_params",base_params)
    #print("data_holder_social_multiplier",data_holder_flow_social_multiplier)
    #print("data_holder_cultural_multiplier",data_holder_flow_cultural_multiplier)
    
    #data_array_social_stock = data_holder_stock_social_multiplier.mean(axis=2)
    #data_array_cultural_stock = data_holder_stock_cultural_multiplier.mean(axis=2)

    #print(data_holder_flow_social_multiplier, data_holder_flow_social_multiplier.shape)
    #quit()
    data_array_social_H = data_holder_flow_social_multiplier.mean(axis=2)
    data_array_cultural_H = data_holder_flow_cultural_multiplier.mean(axis=2)

    #print("vdata_array_social",data_array_social,data_array_social.shape)

    #plot_multipliers(fileName,data_array_social_stock,data_array_cultural_stock, property_values_list, phi_list,"Emissions, E","stock")
    #plot_reduc(fileName,data_array_social_stock,data_array_cultural_stock, property_values_list, phi_list,"Emissions, E","stock")

    plot_multipliers(fileName,data_array_social_H,data_array_cultural_H, property_values_list, phi_list,"Total high carbon consumption, H","H")
    plot_reduc(fileName,data_array_social_H,data_array_cultural_H, property_values_list, phi_list,"Total high carbon consumption, H","H")
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/deriving_multipliers_13_29_04__18_08_2023"
    )

