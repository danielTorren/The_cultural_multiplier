"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from package.resources.utility import load_object
from package.resources.plot import (
    set_latex
)
from matplotlib.cm import get_cmap,rainbow
import numpy as np

def plot_end_points_emissions_scatter(
    fileName: str, Data_list, property_title, property_save, property_vals,dpi_save: int,latex_bool = False 
):
    if latex_bool:
        set_latex()

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6))
    print("Data_list",Data_list)
    print("Data_list.shape", Data_list.shape)
    print(" property_vals", property_vals)

    colors = iter(rainbow(np.linspace(0, 1, Data_list.shape[1])))

    data = Data_list.T

    for i in range(len(Data_list[0])):
        ax.scatter(property_vals,  data[i], color = next(colors))

    ax.set_xlabel(property_title)
    ax.set_ylabel(r"Tax multiplier")

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "_scatter_multiplier"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    dpi_save = 600,
    latex_bool = 0,
    PLOT_TYPE = 1
    ) -> None: 

    ############################

    multiplier_matrix = load_object(fileName + "/Data", "multiplier_matrix")
    property_varied = load_object(fileName + "/Data", "property_varied")
    property_varied_title = load_object(fileName + "/Data", "property_varied_title")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    base_params = load_object(fileName + "/Data", "base_params")
    

    #plot_end_points_emissions(fileName, M_array, "$\mu$", property_varied, property_values_list, dpi_save)
    plot_end_points_emissions_scatter(fileName, multiplier_matrix, "$\mu$", property_varied, property_values_list, dpi_save)
    #plot_end_points_emissions_lines(fileName, M_array, "$\mu$", property_varied, property_values_list, dpi_save)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/emission_target_sweep_15_52_47__05_09_2023"
    )

