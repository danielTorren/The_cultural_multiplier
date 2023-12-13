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
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import seaborn as sns


def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
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

    emissions_array = load_object(fileName + "/Data", "results")
        
    #plot how the emission change for each one
    plot_end_points_emissions(fileName, emissions_array, property_varied_title, property_varied, property_values_list, dpi_save)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/fixed_preferences_16_24_30__06_11_2023",
    )

