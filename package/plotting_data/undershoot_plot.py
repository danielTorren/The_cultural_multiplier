# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from package.resources.utility import ( 
    load_object,
)


def plot_identity_matrix_density_grid(fileName, h_list, scenarios, dpi_save):
    
    fig, axs = plt.subplots(1, len(scenarios), figsize=(12, 12))
    y_title = r"Identity, $I_{t,n}$"
    
    colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
    n_bins = 100  # Discretize the interpolation into bins
    cmap_name = 'white_to_blue'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    for i, scenario in enumerate(scenarios):
        ax = axs[i]
        h = h_list[i]
        X, Y = np.meshgrid(h[1], h[2])
        ax.pcolormesh(X, Y, h[0].T, cmap=custom_cmap, shading='auto')
        ax.set_title(scenario)
        plt.colorbar(ax.pcolormesh(X, Y, h[0].T, cmap=custom_cmap, shading='auto'), ax=ax, label='Density')
    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_identity_timeseries_matrix_density_grid"
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_preferences_matrix_density_grid(fileName, preferneces_h_list, scenarios, dpi_save):
    
    fig, axs = plt.subplots(len(preferneces_h_list), len(scenarios), figsize=(12, 12))
    
    colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
    n_bins = 100  # Discretize the interpolation into bins
    cmap_name = 'white_to_blue'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    for j in range(len(preferneces_h_list)):
        h_list = preferneces_h_list[j]
        for i, scenario in enumerate(scenarios):
            ax = axs[j][i]
            h = h_list[i]
            X, Y = np.meshgrid(h[1], h[2])
            ax.pcolormesh(X, Y, h[0].T, cmap=custom_cmap, shading='auto')
            ax.set_title(scenario)
            plt.colorbar(ax.pcolormesh(X, Y, h[0].T, cmap=custom_cmap, shading='auto'), ax=ax, label='Density')
        axs[j][0].set_xlabel("Preference, $A_{t,n,%s}$" % (j+1))

    fig.supxlabel(r"Time")
    #fig.supylabel(r"%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_preferences_timeseries_matrix_density_grid"
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_stock(fileName, data_stock, scenarios, dpi_save, history_time):
    
    fig, axs = plt.subplots(1, len(scenarios), figsize=(12, 12), sharey=True)
    y_title = "Cumulative carbon emissions, $E_t$"
    
    for i, scenario in enumerate(scenarios):
        ax = axs[i]
        ax.set_title(scenario)
        data_scenario = data_stock[i]
        
        #print(len(history_time),data_scenario.shape )
        for j, data in enumerate(data_scenario):
            ax.plot(history_time, data)

    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_stock_timer"
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_flow(fileName, data_flow, scenarios, dpi_save, history_time):
    
    fig, axs = plt.subplots(1, len(scenarios), figsize=(12, 12), sharey=True)
    y_title = "Flow carbon emissions, $E_{F,t}$"
    
    for i, scenario in enumerate(scenarios):
        ax = axs[i]
        ax.set_title(scenario)
        data_scenario = data_flow[i]

        for j, data in enumerate(data_scenario):
            ax.plot(history_time, data)

    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_flow_timer"
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_final_values_histogram(fileName, data_stock, scenarios, dpi_save):
    
    fig, axs = plt.subplots(1, len(scenarios), figsize=(12, 6), sharey=True)
    y_title = "Frequency"
    
    for i, scenario in enumerate(scenarios):
        ax = axs[i]
        ax.set_title(scenario)
        
        # Extract final values from each time series in the scenario
        final_values = [data[-1] for data in data_stock[i]]
        
        # Plot histogram of final values
        ax.hist(final_values, bins=100, color='blue', alpha=0.7)
        
        ax.set_xlabel("Final cumulative emissions, $E_{t_{max}}$")
        if i == 0:
            ax.set_ylabel(y_title)
    
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/histogram_final_values"
    fig.savefig(f + ".png", dpi=dpi_save, format="png")
    plt.show()

def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    dpi_save = 600,
    ) -> None: 

    #data_array = load_object(fileName + "/Data", "data_array")
    h_list = load_object(fileName + "/Data", "h_list")
    preferences_h_list = load_object(fileName + "/Data", "h_list_preferences_sectors")
    data_flow = load_object(fileName + "/Data", "data_flow")
    data_stock = load_object(fileName + "/Data", "data_stock")
    history_time = load_object(fileName + "/Data", "history_time")
    #scenarios = load_object(fileName + "/Data", "scenarios")
    scenarios = ["fixed_preferences","dynamic_socially_determined_weights", "dynamic_identity_determined_weights" ]

    plot_identity_matrix_density_grid(fileName, h_list, scenarios, dpi_save)
    plot_preferences_matrix_density_grid(fileName, preferences_h_list, scenarios, dpi_save)
    plot_stock(fileName, data_stock, scenarios, dpi_save, history_time)
    plot_flow(fileName, data_flow, scenarios, dpi_save, history_time)
    plot_final_values_histogram(fileName, data_stock, scenarios, dpi_save)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/undershoot_13_14_23__20_08_2024",
    )


