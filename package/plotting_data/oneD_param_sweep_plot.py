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

def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    dpi_save = 600,
    latex_bool = 0,
    PLOT_TYPE = 1
    ) -> None: 

    ############################

    emissions_array = load_object(fileName + "/Data", "emissions_array")
    property_varied = load_object(fileName + "/Data", "property_varied")
    property_varied_title = load_object(fileName + "/Data", "property_varied_title")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    base_params = load_object(fileName + "/Data", "base_params")
    
    #print(base_params)
    #quit()

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
    else:
        plot_end_points_emissions(fileName, emissions_array, property_varied_title, property_varied, property_values_list, dpi_save)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/one_param_sweep_multi_15_57_44__24_05_2023",
        PLOT_TYPE = 4
    )

