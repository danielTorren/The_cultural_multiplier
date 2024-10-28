import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import load_object

import matplotlib.pyplot as plt
import numpy as np

def plot_emissions_scatter(variable_parameters_dict, emissions_data):
    """
    Plot emissions as a scatter plot with M on the x-axis, M_identity on the y-axis,
    and emissions represented by color.

    Parameters:
        variable_parameters_dict (dict): Dictionary containing valid (M, M_identity) combinations.
        emissions_data (list or numpy.ndarray): Emissions values for each (M, M_identity) pair.
    """
    # Extract M and M_identity values from variable_parameters_dict
    M_M_identity_combinations = variable_parameters_dict["M_M_identity_combinations"]
    M_values = [pair[0] for pair in M_M_identity_combinations]
    M_identity_values = [pair[1] for pair in M_M_identity_combinations]
    #print(emissions_data.shape)
    #quit()
    emissions_data_reduc =  emissions_data.reshape(len(M_identity_values), 5 )
    emissions_data = np.mean(emissions_data_reduc, axis = 1)
    print("emissions_data", emissions_data)
    print("M_M_identity_combinations", M_M_identity_combinations)
    plt.figure(figsize=(10, 8))
    
    # Scatter plot where color represents emissions
    scatter = plt.scatter(M_values, M_identity_values, c=emissions_data, cmap="viridis", s=100, edgecolor='k')
    plt.colorbar(scatter, label='Average Emissions')
    
    plt.xlabel("M")
    plt.ylabel("M_identity")
    plt.title("Emissions Scatter Plot by M and M_identity")
    plt.show()

def main(
    fileName
) -> None:
    
    #FULL
    emissions_data = load_object(fileName + "/Data","emissions_data") 
    variable_parameters_dict = load_object(fileName + "/Data","variable_parameters_dict")  

    plot_emissions_scatter(variable_parameters_dict, emissions_data)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/SW_M_M_identity_20_00_26__28_10_2024"#tax_sweep_networks_16_44_43__18_09_2024",#tax_sweep_networks_15_57_56__22_08_2024",
    )