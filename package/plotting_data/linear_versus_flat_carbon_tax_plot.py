"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from package.resources.plot import (
    plot_emissions_flat_versus_linear
)

def main(
    fileName = "results/test",
    ) -> None: 
    ############################

    data_flat = load_object(fileName + "/Data", "data_flat")
    data_linear = load_object(fileName + "/Data", "data_linear")
    carbon_prices = load_object(fileName + "/Data", "property_values_list")

    #plot how the emission change for each one
    plot_emissions_flat_versus_linear(fileName, data_flat,data_linear, carbon_prices)

    plt.show()
if __name__ == '__main__':
    plots = main(
        fileName="results/flat_linear_sweep_carbon_price_15_19_05__23_05_2023"
    )