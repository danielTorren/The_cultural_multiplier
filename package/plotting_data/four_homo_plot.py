"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from package.resources.utility import ( 
    load_object,
)

def plot_identity_matrix_density_four(fileName, data_list, dpi_save, bin_num, latex_bool=False):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    y_title = r"Identity, $I_{t,n}$"
    
    colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
    n_bins = 100  # Discretize the interpolation into bins
    cmap_name = 'white_to_blue'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    titles = [
        "Homophily 0, Coherence 0",
        "Homophily 0, Coherence 1",
        "Homophily 1, Coherence 0",
        "Homophily 1, Coherence 1"
    ]
    
    for i, data in enumerate(data_list):
        ax = axs[i // 2, i % 2]
        
        # Transpose the data to get time steps on the x-axis and identity values on the y-axis
        Data_preferences_trans = np.asarray(data.history_identity_vec).T  # Now it's time then person
        
        # Create a 2D histogram
        h = ax.hist2d(np.tile(np.asarray(data.history_time), data.N), Data_preferences_trans.flatten(), bins=[len(data.history_time), bin_num], cmap=custom_cmap)
        
        # Set the labels
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(titles[i])
        
        # Add color bar to indicate the density
        plt.colorbar(h[3], ax=ax, label='Density')
    
    #plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_identity_timeseries_matrix_density_four"
    fig.savefig(f + ".png", dpi=dpi_save, format="png")
    
def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    dpi_save = 600,
    ) -> None: 

    Data_list = load_object(fileName + "/Data", "Data_list")

    #print("Data EMissions", Data.total_carbon_emissions_stock)

    #node_shape_list = ["o","s","^","v"]

    #anim_save_bool = False#Need to install the saving thing
    ###PLOTS

    
    #plot_identity_matrix(fileName, Data, dpi_save)
    bin_num= 200
    plot_identity_matrix_density_four(fileName, Data_list, dpi_save, bin_num)


    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/homo_four_19_19_42__06_08_2024",
    )


