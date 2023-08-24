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

    M_array = load_object(fileName + "/Data", "multiplier_matrix")
    property_varied = load_object(fileName + "/Data", "property_varied")
    property_varied_title = load_object(fileName + "/Data", "property_varied_title")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    base_params = load_object(fileName + "/Data", "base_params")
    

    plot_end_points_emissions(fileName, M_array, "$\mu$", property_varied, property_values_list, dpi_save)
    plot_end_points_emissions_scatter(fileName, M_array, "$\mu$", property_varied, property_values_list, dpi_save)
    plot_end_points_emissions_lines(fileName, M_array, "$\mu$", property_varied, property_values_list, dpi_save)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/emission_target_sweep_19_39_26__24_08_2023"
    )

