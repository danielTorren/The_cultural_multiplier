# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from package.resources.utility import ( 
    load_object,
)


def plot_identity_matrix_density_four_array(fileName, data_array, dpi_save, bin_num, latex_bool=False):
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
    
    time_steps,N, = data_array.shape[1], data_array.shape[2]
    history_time = np.arange(time_steps)#np.linspace(0, 10, time_steps)  # Assuming the same time steps for all data
    time_tile = np.tile(history_time, N)

    for i in range(4):
        ax = axs[i // 2, i % 2]
        # Extract the specific dataset
        data = data_array[i].T
        data_flat = data.flatten()
        h = ax.hist2d(time_tile, data_flat, bins=[time_steps, bin_num], cmap=custom_cmap)
        
        # Set the labels
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(titles[i])
        
        # Add color bar to indicate the density
        plt.colorbar(h[3], ax=ax, label='Density')
    
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_identity_timeseries_matrix_density_four"
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_identity_matrix_density_four_array_seed(fileName, data_array, dpi_save, bin_num, latex_bool=False):
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
    
    seeds, time_steps, N = data_array.shape[1], data_array.shape[2], data_array.shape[3]
    history_time = np.arange(time_steps)  # Assuming the same time steps for all data
    time_tile = np.tile(history_time, N * seeds)
    
    for i in range(4):
        ax = axs[i // 2, i % 2]
        data_subfigure = data_array[i]
        data_trans = data_subfigure.transpose(0, 2, 1)
        combined_data = data_trans.reshape(seeds * N, time_steps)
        data_flat = combined_data.flatten()

        h = ax.hist2d(time_tile, data_flat, bins=[time_steps, bin_num], cmap=custom_cmap)
        
        # Set the labels
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(titles[i])
        
        # Add color bar to indicate the density
        plt.colorbar(h[3], ax=ax, label='Density')
    
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_identity_timeseries_matrix_density_four_array_seed"
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    dpi_save = 600,
    ) -> None: 

    data_array = load_object(fileName + "/Data", "data_array")

    bin_num= 50
    #plot_identity_matrix_density_four_array(fileName, data_array, dpi_save, bin_num)
    plot_identity_matrix_density_four_array_seed(fileName, data_array, dpi_save, bin_num)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/homo_four_11_49_11__07_08_2024",
    )


