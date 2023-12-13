"""
"""

# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from package.resources.plot import (
    plot_emissions_timeseries
)
import numpy as np

def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    dpi_save = 600,
    ) -> None: 

    ############################

    emissions_array = load_object(fileName + "/Data", "emissions_array")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    base_params = load_object(fileName + "/Data", "base_params")
    time_array = np.arange(0,base_params["time_steps_max"] + base_params["compression_factor_state"],base_params["compression_factor_state"])

    plot_emissions_timeseries(fileName, emissions_array, property_values_list, time_array)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/stochastic_sweep_18_14_41__28_04_2023",
    )

