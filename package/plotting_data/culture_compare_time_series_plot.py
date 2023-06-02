"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from package.resources.utility import load_object
from package.resources.plot import (
    plot_low_carbon_preferences_timeseries_compare_culture,
    plot_emissions_timeseries_compare_culture,
    plot_stock_emissions_timeseries_compare_culture,
    plot_identity_timeseries_compare_culture,
    plot_flow_emissions_timeseries_compare_culture
)

def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    dpi_save = 600,
    ) -> None: 

    ############################

    Data_list = load_object(fileName + "/Data", "Data_list")
    base_params = load_object(fileName + "/Data", "base_params")
    culture_list = load_object(fileName + "/Data", "culture_list")
    #culture_list = ["Culture", "No culture", "Static preferences"]

    plot_low_carbon_preferences_timeseries_compare_culture(fileName, Data_list, dpi_save,culture_list)
    plot_emissions_timeseries_compare_culture(fileName, Data_list, dpi_save,culture_list)
    plot_identity_timeseries_compare_culture(fileName, Data_list, dpi_save,culture_list)
    plot_stock_emissions_timeseries_compare_culture(fileName, Data_list, dpi_save,culture_list)
    plot_flow_emissions_timeseries_compare_culture(fileName, Data_list, dpi_save,culture_list)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/culture_compare_time_series_20_37_19__30_05_2023",
    )

