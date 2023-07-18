"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
import numpy as np
from package.resources.utility import ( 
    load_object,
)
from package.resources.plot import (
    plot_identity_timeseries,
    plot_value_timeseries,
    plot_attitude_timeseries,
    plot_total_carbon_emissions_timeseries,
    plot_weighting_matrix_convergence_timeseries,
    plot_cultural_range_timeseries,
    plot_average_identity_timeseries,
    plot_joint_cluster_micro,
    print_live_initial_identity_network,
    live_animate_identity_network_weighting_matrix,
    plot_low_carbon_preferences_timeseries,
    plot_total_flow_carbon_emissions_timeseries
)


def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    PLOT_NAME = "INDIVIDUAL",
    node_size = 50,
    fps = 5,
    interval = 50,
    layout = "circular",
    round_dec = 3,
    dpi_save = 2000,
    no_samples = 10000,
    bandwidth = 0.01,
    shuffle_colours = True,
    animation_save_bool = 0,
    latex_bool = 0
    ) -> None: 

    cmap_multi = get_cmap("plasma")
    cmap_weighting = get_cmap("Reds")

    norm_zero_one = Normalize(vmin=0, vmax=1)
    cmap = LinearSegmentedColormap.from_list(
        "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
    )

    Data = load_object(fileName + "/Data", "social_network")

    ###PLOTS
    plot_low_carbon_preferences_timeseries(fileName, Data, dpi_save)
    plot_identity_timeseries(fileName, Data, dpi_save)
    plot_total_carbon_emissions_timeseries(fileName, Data, dpi_save,latex_bool = latex_bool)
    plot_total_flow_carbon_emissions_timeseries(fileName, Data, dpi_save,latex_bool = latex_bool)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/single_experiment_19_12_36__04_05_2023"
    )


