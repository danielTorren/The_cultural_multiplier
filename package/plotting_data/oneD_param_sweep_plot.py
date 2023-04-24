"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from package.resources.utility import load_object
from package.resources.plot import (
    plot_end_points_emissions

)

def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    dpi_save = 600,
    latex_bool = 0
    ) -> None: 

    ############################

    emissions_array = load_object(fileName + "/Data", "emissions_array")
    property_varied = load_object(fileName + "/Data", "property_varied")
    property_varied_title = load_object(fileName + "/Data", "property_varied_title")
    property_values_list = load_object(fileName + "/Data", "property_values_list")

    #plot how the emission change for each one
    plot_end_points_emissions(fileName, emissions_array, property_varied_title, property_varied, property_values_list , dpi_save)

    plt.show()
if __name__ == '__main__':
    plots = main(
        fileName="results/one_param_sweep_multi_21_58_44__24_04_2023"
    )

