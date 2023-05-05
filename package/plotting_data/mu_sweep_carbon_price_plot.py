"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from package.resources.plot import (
    multi_emissions_timeseries_carbon_price,
    multi_emissions_timeseries_carbon_price_quantile
)
from package.resources.utility import (
    load_object
)

def main(
    fileName = "results/mu_sweep_carbon_price_11_48_09__05_05_2023",
) -> None:
        
    emissions_stock_data_list = load_object(fileName + "/Data","emissions_stock_data_list")
    #print("emissions_stock_data_list", emissions_stock_data_list.shape)
    emissions_flow_data_list = load_object(fileName + "/Data","emissions_flow_data_list")
    carbon_prices_dict = load_object(fileName + "/Data", "carbon_prices") 
    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    time_array = load_object(fileName + "/Data", "time_array") 
    #multi_emissions_timeseries_carbon_price(fileName,emissions_stock_data_list, carbon_prices_dict["carbon_prices"] ,property_values_list,time_array,"Carbon emissions stock", "stock")
    #multi_emissions_timeseries_carbon_price(fileName,emissions_flow_data_list, carbon_prices_dict["carbon_prices"] ,property_values_list,time_array,"Carbon emissions flow","flow")
    
    multi_emissions_timeseries_carbon_price_quantile(fileName,emissions_stock_data_list, carbon_prices_dict["carbon_prices"] ,property_values_list,time_array,"Carbon emissions stock", "stock")
    multi_emissions_timeseries_carbon_price_quantile(fileName,emissions_flow_data_list, carbon_prices_dict["carbon_prices"] ,property_values_list,time_array,"Carbon emissions flow","flow")
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/mu_sweep_carbon_price_13_45_53__05_05_2023",
    )