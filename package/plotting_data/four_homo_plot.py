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

def plot_identity_matrix_density_four(fileName, Data, dpi_save, bin_num, latex_bool=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_title = r"Identity, $I_{t,n}$"


    colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
    n_bins = 100  # Discretize the interpolation into bins
    cmap_name = 'white_to_blue'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Transpose the data to get time steps on the x-axis and identity values on the y-axis
    Data_preferences_trans = np.asarray(Data.history_identity_vec).T  # Now it's time then person

    # Create a 2D histogram
    h = ax.hist2d(np.tile(np.asarray(Data.history_time), Data.N), Data_preferences_trans.flatten(), bins=[len(Data.history_time), bin_num],cmap = custom_cmap)# cmap='viridis')

    # Set the labels
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    # Add color bar to indicate the density
    plt.colorbar(h[3], ax=ax, label='Density')

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_identity_timeseries_matrix_density"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    dpi_save = 600,
    ) -> None: 

    Data = load_object(fileName + "/Data", "social_network")

    #print("Data EMissions", Data.total_carbon_emissions_stock)

    #node_shape_list = ["o","s","^","v"]

    #anim_save_bool = False#Need to install the saving thing
    ###PLOTS

    
    #plot_identity_matrix(fileName, Data, dpi_save)
    bin_num= 200
    plot_identity_matrix_density(fileName, Data, dpi_save, bin_num)
    #plot_emissions_flow_matrix(fileName, Data, dpi_save)
    #plot_emissions_individuals(fileName, Data, dpi_save)
    #plot_identity_timeseries(fileName, Data, dpi_save)

    """
    if Data.burn_in_duration == 0:
        plot_consumption_no_burn_in(fileName, Data, dpi_save)
    else:
        plot_consumption(fileName, Data, dpi_save)
    """
    #print("Data eminissions",Data.total_carbon_emissions_stock )
    #quit()
    #print(Data.parameters)
    #quit()

    

    #plot_total_carbon_emissions_timeseries(fileName, Data, dpi_save)
    #plot_total_flow_carbon_emissions_timeseries(fileName, Data, dpi_save)
    #plot_chi(fileName, Data, dpi_save)
    #plot_omega(fileName, Data, dpi_save)
    #plot_consum_ratio(fileName, Data, dpi_save)
    #plot_L(fileName, Data, dpi_save)
    #plot_H(fileName, Data, dpi_save)
    #plot_Z_timeseries(fileName, Data, dpi_save)

    """
    if Data.network_type =="SBM":
        plot_SBM_low_carbon_preferences_timeseries(fileName, Data, dpi_save)
        plot_SBM_network_start_preferences(fileName, Data,cmap, dpi_save, node_sizes,norm_zero_one,block_markers_list,legend_loc,lines_alpha)
        plot_SBM_network_end_preferences(fileName, Data,cmap, dpi_save, node_sizes,norm_zero_one,block_markers_list,legend_loc,lines_alpha)
    else:
        plot_low_carbon_preferences_timeseries(fileName, Data, dpi_save)
        plot_network_start_preferences(fileName, Data,cmap, dpi_save, node_sizes,norm_zero_one)
        plot_network_end_preferences(fileName, Data,cmap, dpi_save, node_sizes,norm_zero_one)
    """
    #threshold_list = [0.0001,0.0002,0.0005,0.001,0.002,0.003,0.004]
    #emissions_threshold_range = np.arange(0,0.005,0.000001)
    #plot_low_carbon_adoption_timeseries(fileName, Data,threshold_list, dpi_save)
    #plot_low_carbon_adoption_transition_timeseries(fileName, Data,emissions_threshold_range, dpi_save)
    
    #anim_1 = variable_animation_distribution(fileName, Data, "history_identity","Identity", dpi_save,anim_save_bool)
    #anim_2 = varaible_animation_distribution(fileName, Data, "history_flow_carbon_emissions","Individual emissions" , dpi_save,anim_save_bool)

    #anim_3 = fixed_animation_distribution(fileName, Data, "history_identity","Identity","y", dpi_save,anim_save_bool)
    #anim_4 = fixed_animation_distribution(fileName, Data, "history_flow_carbon_emissions","Individual emissions","y",dpi_save,anim_save_bool)
    #anim_5 = fixed_animation_distribution(fileName, Data, "history_utility","Utility","y",dpi_save,anim_save_bool)
    
    #VECTORS
    #anim_4 = multi_col_fixed_animation_distribution(fileName, Data, "history_low_carbon_preferences","Low carbon Preferences","y", dpi_save,anim_save_bool)


    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/single_experiment_18_55_45__06_08_2024",
    )


