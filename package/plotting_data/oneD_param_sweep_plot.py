"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from package.resources.utility import load_object
from package.resources.plot import (
    plot_end_points_emissions,
    plot_end_points_emissions_scatter,
    plot_end_points_emissions_lines,
    plot_end_points_emissions_scatter_gini,
    plot_end_points_emissions_lines_gini
)
import numpy as np

def plot_stacked_preferences(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(data_list),ncols=data_list[0].M, sharey="row", sharex="col")

    for i, data in enumerate(data_list):
        axes[i][0].set_title(property_varied_title + " = " + str(property_values_list[i]))
        for v in range(data.N):
            data_indivdiual = np.asarray(data.agent_list[v].history_low_carbon_preferences)
            for j in range(data.M):
                axes[i][j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )

    fig.supxlabel(r"Time")
    fig.supylabel(r"Low carbon preference")

    plotName = fileName + "/Prints"

    f = plotName + "/timeseries_preference_stacked_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    dpi_save = 600,
    latex_bool = 0,
    PLOT_TYPE = 1
    ) -> None: 

    ############################
    
    #print(base_params)
    #quit()

    base_params = load_object(fileName + "/Data", "base_params")
    var_params  = load_object(fileName + "/Data" , "var_params")
    property_values_list = load_object(fileName + "/Data", "property_values_list")

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption",
    property_min = var_params["property_min"]#0,
    property_max = var_params["property_max"]#1,
    property_reps = var_params["property_reps"]#10,
    property_varied_title = var_params["property_varied_title"]

    if PLOT_TYPE == 5:
        data_list = load_object(fileName + "/Data", "data_list")
    else:
        emissions_array = load_object(fileName + "/Data", "emissions_array")
        
    if PLOT_TYPE == 1:
        reduc_emissions_array = emissions_array[:-1]
        reduc_property_values_list = property_values_list[:-1]
        #plot how the emission change for each one
        plot_end_points_emissions(fileName, reduc_emissions_array, property_varied_title, property_varied, reduc_property_values_list, dpi_save)
    elif PLOT_TYPE == 2:
        plot_end_points_emissions(fileName, emissions_array, "Preference to consumption ratio, $\\mu$", property_varied, property_values_list, dpi_save)
    elif PLOT_TYPE == 3:
        gini_array = load_object(fileName + "/Data", "gini_array")

        plot_end_points_emissions(fileName, emissions_array, "Budget inequality (Pareto distribution constant)", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_scatter_gini(fileName, emissions_array, "Initial Gini index", property_varied, property_values_list,gini_array, dpi_save)
        plot_end_points_emissions_lines_gini(fileName, emissions_array, "Initial Gini index", property_varied, property_values_list,gini_array, dpi_save)
    elif PLOT_TYPE == 4:
        #gini_array = load_object(fileName + "/Data", "gini_array")
        plot_end_points_emissions(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_scatter(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_lines(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
    if PLOT_TYPE == 5:
        # look at splitting of the last behaviour with preference dissonance
        plot_stacked_preferences(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
    else:
        plot_end_points_emissions(fileName, emissions_array, property_varied_title, property_varied, property_values_list, dpi_save)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/one_param_sweep_multi_13_16_23__25_10_2023",
        PLOT_TYPE = 5
    )

