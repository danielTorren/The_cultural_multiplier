"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import load_object

def plot_bifurcation_sectors(fileName, data_array,base_params,property_values_list, property_varied, property_varied_title, dpi_save):

    if base_params["M"] == 1:
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout = True, figsize= (10,6))

        #I need to sperate the data out so that its shape is: sector(M), one d thing im varying (Q?), the different people(N)[M,Q,N], currently its (Q,N,M)
        transposed_data = np.transpose(data_array, (2, 0, 1))
        data_sector = transposed_data[0]
        for j in range(len(property_values_list)):
            x_vals = [property_values_list[j]]*(len(data_sector[j]))
            y_vals = data_sector[j]
            ax.plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)

        ax.set_ylabel("Final preference, $A_{t_{max},i,%s}$" % (str(1)))
        ax.set_xlabel(property_varied_title)
        ax.set_ylim(0,1)

    else:
        fig, axes = plt.subplots(nrows=1, ncols=base_params["M"],sharey="row", constrained_layout = True, figsize= (10,6))

        #I need to sperate the data out so that its shape is: sector(M), one d thing im varying (Q?), the different people(N)[M,Q,N], currently its (Q,N,M)
        transposed_data = np.transpose(data_array, (2, 0, 1))

        
        for i, ax in enumerate(axes.flat):
            data_sector = transposed_data[i]
            for j in range(len(property_values_list)):
                x_vals = [property_values_list[j]]*(len(data_sector[j]))
                y_vals = data_sector[j]
                ax.plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)

            ax.set_ylabel("Final preference, $A_{\\tau,i,%s}$" % (str(i+1)))
            ax.set_xlabel(property_varied_title)
            ax.set_ylim(0,1)

    plotName = fileName + "/Prints"

    f = plotName + "/bifurcation_preferences_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_bifurcation_sectors_log(fileName, data_array,base_params,property_values_list, property_varied, property_varied_title, dpi_save):

    if base_params["M"] == 1:
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout = True, figsize= (10,6))

        #I need to sperate the data out so that its shape is: sector(M), one d thing im varying (Q?), the different people(N)[M,Q,N], currently its (Q,N,M)
        transposed_data = np.transpose(data_array, (2, 0, 1))
        data_sector = transposed_data[0]
        for j in range(len(property_values_list)):
            x_vals = [property_values_list[j]]*(len(data_sector[j]))
            y_vals = data_sector[j]
            ax.plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)

        ax.set_ylabel("Final preference, $A_{t_{max},i,%s}$" % (str(1)))
        ax.set_xlabel(property_varied_title)
        ax.set_ylim(0,1)
        ax.set_xscale('log')

    else:
        fig, axes = plt.subplots(nrows=1, ncols=base_params["M"],sharey="row", constrained_layout = True, figsize= (10,6))

        #I need to sperate the data out so that its shape is: sector(M), one d thing im varying (Q?), the different people(N)[M,Q,N], currently its (Q,N,M)
        transposed_data = np.transpose(data_array, (2, 0, 1))

        
        for i, ax in enumerate(axes.flat):
            data_sector = transposed_data[i]
            for j in range(len(property_values_list)):
                x_vals = [property_values_list[j]]*(len(data_sector[j]))
                y_vals = data_sector[j]
                ax.plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)

            ax.set_ylabel("Final preference, $A_{\\tau,i,%s}$" % (str(i+1)))
            ax.set_xlabel(property_varied_title)
            ax.set_ylim(0,1)
            ax.set_xscale('log')

    plotName = fileName + "/Prints"

    f = plotName + "/log_bifurcation_preferences_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def main(
    fileName = "results/one_param_sweep_multi_12_01_46__31_10_2023",
    dpi_save = 600,
    latex_bool = 0,
    PLOT_TYPE = 1
    ) -> None: 

    ############################
    
    base_params = load_object(fileName + "/Data", "base_params")
    var_params  = load_object(fileName + "/Data" , "var_params")
    property_values_list = load_object(fileName + "/Data", "property_values_list")

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
    property_min = var_params["property_min"]#0,
    property_max = var_params["property_max"]#1,    
    property_reps = var_params["property_reps"]#10,
    property_varied_title = var_params["property_varied_title"]

    data_array = load_object(fileName + "/Data", "data_array")

    if PLOT_TYPE == 1:
        # look at splitting of the last behaviour with preference dissonance at final time step
        #"$\\sigma$ upper"
        #plot_bifurcation_sectors(fileName,data_array,base_params,property_values_list, property_varied, property_varied_title, dpi_save)
        plot_bifurcation_sectors(fileName,data_array,base_params,property_values_list, property_varied, "$\\sigma_1$", dpi_save)
        plot_bifurcation_sectors_log(fileName,data_array,base_params,property_values_list, property_varied, property_varied_title, dpi_save)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/2D_bifur_param_sweep_14_52_59__04_12_2023",
        PLOT_TYPE = 1
    )

