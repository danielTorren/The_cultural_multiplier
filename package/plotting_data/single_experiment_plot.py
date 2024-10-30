"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import ( 
    load_object,
)
import networkx as nx

def plot_identity_matrix(fileName, Data, dpi_save,latex_bool = False):

    fig, ax = plt.subplots(figsize=(10,6))
    y_title = r"Identity, $I_{t,n}$"

    Data_preferences_trans = np.asarray(Data.history_identity_vec).T#NOW ITS person then time

    for v in range(Data.N):
        ax.plot(np.asarray(Data.history_time), Data_preferences_trans[v])
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        #ax.set_ylim(0, 1)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_identity_timeseries_matrix"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")


def plot_preference_timeseries(fileName, Data, dpi_save, latex_bool=False):

    if latex_bool:
        plt.rcParams['text.usetex'] = True
    else:
        plt.rcParams['text.usetex'] = False

    fig, axes = plt.subplots(nrows=1, ncols = Data.M,figsize=(10, 6))
    
    y_title = r"Low Carbon Preference, $A_{t,i,m}$"

    # Transpose the preference history matrix to have individuals as rows and time as columns

    Data_preferences_trans = np.asarray(Data.history_low_carbon_preference_matrix).T
    # Plot each individual's preference evolution over time
    for m in range(Data.M):
        ax = axes[m]
        for n in range(Data.N):
            ax.plot(np.asarray(Data.history_time), Data_preferences_trans[m][n])
            ax.set_xlabel(r"Time")
            ax.set_title("m = %s" % (m+1))
            #ax.set_ylabel(y_title)
            # Optionally set y-axis limits if needed
            # ax.set_ylim(0, 1)
    axes[0].set_ylabel(y_title)
    plt.tight_layout()

    # Save the plots in the specified formats
    plotName = fileName + "/Plots"
    f = plotName + "/plot_preference_timeseries_matrix"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_network(network_obj):
    """
    Plot the social network with nodes colored by identity values at the initial and final time steps.
    
    Args:
        network_obj (Network_Matrix): Instance of the Network_Matrix class
    """
    G = network_obj.network
    
    # Get initial and final identity values
    initial_identity = network_obj.history_identity_vec[0]
    final_identity = network_obj.history_identity_vec[-1]
    
    # Create a colormap based on the identity values
    cmap = plt.get_cmap('viridis')
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the initial network
    pos = nx.spring_layout(G)
    node_colors = [cmap(v) for v in initial_identity]
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=3, ax=ax1)
    ax1.set_title("Initial Network")
    
    # Plot the final network
    node_colors = [cmap(v) for v in final_identity]
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=3, ax=ax2)
    ax2.set_title("Final Network")
    
    # Add a shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = fig.colorbar(sm, ax=[ax1, ax2], location='right', pad=0.15)
    cbar.set_label("Environmental identity, I")  # Add label here
    fig.suptitle("Social Network Evolution")

def plot_total_carbon_emissions_timeseries(
    fileName: str, Data, dpi_save: int,latex_bool = False
):
    y_title = "Carbon Emissions Stock"
    property = "history_stock_carbon_emissions"
    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_total_flow_carbon_emissions_timeseries(
    fileName: str, Data, dpi_save: int,latex_bool = False
):

    y_title = "Carbon Emissions Flow"
    property = "history_flow_carbon_emissions"
    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)

def plot_network_timeseries(
    fileName: str, Data: Network, y_title: str, property: str, dpi_save: int,latex_bool = False
):
    fig, ax = plt.subplots(figsize=(10,6))
    data = eval("Data.%s" % property)

    # bodge
    ax.plot(Data.history_time, data)
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    fig.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/" + property + "_timeseries"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    dpi_save = 600,
    ) -> None: 

    Data = load_object(fileName + "/Data", "social_network")

    plot_network(Data)
    plot_identity_matrix(fileName, Data, dpi_save)
    plot_preference_timeseries(fileName, Data, dpi_save)
    plot_total_carbon_emissions_timeseries(fileName, Data, dpi_save)
    plot_total_flow_carbon_emissions_timeseries(fileName, Data, dpi_save)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/single_experiment_12_00_30__28_08_2024",
    )


