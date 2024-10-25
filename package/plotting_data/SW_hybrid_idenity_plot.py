import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import load_object

import matplotlib.pyplot as plt
import numpy as np

def plot_emissions(data_serial, variable_parameters_dict):
    # Reshape the data as required and calculate the mean across the 'seed_reps' axis
    data_mean = data_serial.reshape(
        variable_parameters_dict["row"]["property_reps"],
        variable_parameters_dict["col"]["property_reps"],
        -1
    ).mean(axis=-1)

    # Define the M and M_identity range (assuming these are sequential or predefined)
    M_values = np.arange(data_mean.shape[1])
    M_identity_values = np.arange(data_mean.shape[0])

    # Create a meshgrid for M and M_identity
    M_grid, M_identity_grid = np.meshgrid(M_values, M_identity_values)

    # Plot the emissions data as a single heatmap
    plt.figure(figsize=(10, 8))
    color_plot = plt.pcolormesh(M_grid, M_identity_grid, data_mean, shading='auto', cmap='viridis')
    
    # Add labels and color bar
    plt.xlabel("M")
    plt.ylabel("Proportion cultural multiplier")
    plt.colorbar(color_plot, label='Emissions')
    plt.title("Mean Emissions Across Seed Reps")

    plt.show()



def main(
    fileName
) -> None:
    
    #FULL
    emissions_data = load_object(fileName + "/Data","emissions_data") 
    print(emissions_data.shape)
    variable_parameters_dict = load_object(fileName + "/Data","variable_parameters_dict")  

    plot_emissions(emissions_data , variable_parameters_dict)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/SW_M_M_identity_18_37_08__25_10_2024"#tax_sweep_networks_16_44_43__18_09_2024",#tax_sweep_networks_15_57_56__22_08_2024",
    )