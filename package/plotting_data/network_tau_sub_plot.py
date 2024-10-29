import matplotlib.pyplot as plt
from package.resources.utility import (
    load_object,
    calc_bounds
)
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np

def plot_means_end_points_emissions(
    fileName, emissions_networks, property_values_list_col, property_values_list_row, network_titles, row_titles, colors
):

    ncols = 3 
    nrows = 1
    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(15,6), constrained_layout = True)# 

    for j in range(ncols):
        axes[j].set_title(network_titles[j])
        for k in range(len(property_values_list_row)):
            axes[j].grid()
            Data = emissions_networks[j][k]
            #print("Data", Data.shape)
            mu_emissions =  Data.mean(axis=1)
            axes[j].plot(property_values_list_col, mu_emissions, label=row_titles[k], c = colors[k])

            Data_trans = Data.T
            for v, data_run in enumerate(Data_trans):
                axes[j].plot(property_values_list_col, data_run, c = colors[k], alpha = 0.1)

        
    fig.supxlabel(r"Carbon tax, $\tau$")
    fig.supylabel(r"Cumulative carbon emissions, E")

    axes[2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_sub_tau_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_means_end_points_emissions_confidence(
    fileName, emissions_networks, property_values_list_col, property_values_list_row, network_titles, row_titles, colors
):

    ncols = 3 
    nrows = 1
    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(12,6), constrained_layout = True)# 

    for j in range(ncols):
        axes[j].set_title(network_titles[j], fontsize = "12")
        for k in range(len(property_values_list_row)):
            axes[j].grid()
            Data = emissions_networks[j][k]
            #print("Data", Data.shape)
            mu_emissions =  Data.mean(axis=1)
            axes[j].plot(property_values_list_col, mu_emissions, label=row_titles[k], c = colors[k])

            mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
            # Plot the 95% confidence interval as a shaded area
            axes[j].fill_between(property_values_list_col, lower_bound, upper_bound, color=colors[k], alpha=0.3)

        
    fig.supxlabel(r"Carbon tax, $\tau$", fontsize = "12")
    fig.supylabel(r"Cumulative carbon emissions, E", fontsize = "12")

    axes[2].legend( fontsize="8")
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/network_sub_tau_emissions_confidence"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_means_end_points_emissions_confidence_split(
    fileName, emissions_networks, property_values_list_col, property_values_list_row, network_titles, row_titles, colors
):
    
    name = "viridis"#"Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors= cmap.colors  # type: list

    # First figure: Small-world network
    ncols_sw = 1  # We only want 1 column for the small-world network
    fig_sw, ax_sw = plt.subplots(ncols=ncols_sw, nrows=1, figsize=(6,6), constrained_layout=True)

    ax_sw.set_title(network_titles[0], fontsize="12")  # Title for small-world network
    for k in range(len(property_values_list_row)):
        ax_sw.grid()
        Data = emissions_networks[0][k]
        mu_emissions = Data.mean(axis=1)
        ax_sw.plot(property_values_list_col, mu_emissions, label=row_titles[k], c=colors[k])

        mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
        # Plot the 95% confidence interval as a shaded area
        ax_sw.fill_between(property_values_list_col, lower_bound, upper_bound, color=colors[k], alpha=0.3)

    fig_sw.supxlabel(r"Carbon tax, $\tau$", fontsize="12")
    fig_sw.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    ax_sw.legend(fontsize="8")

    # Save the small-world network figure
    plotName_sw = fileName + "/Plots"
    f_sw = plotName_sw + "/small_world_tau_emissions_confidence"
    fig_sw.savefig(f_sw + ".png", dpi=600, format="png")
    
    # Second figure: Stochastic Block Model and Scale-free networks
    ncols_other = 2  # 2 columns for Stochastic Block Model and Scale-free networks
    fig_other, axes_other = plt.subplots(ncols=ncols_other, nrows=1, figsize=(12,6), constrained_layout=True)

    for j in range(1, 3):  # Loop over the other two networks
        axes_other[j-1].set_title(network_titles[j], fontsize="12")
        for k in range(len(property_values_list_row)):
            axes_other[j-1].grid()
            Data = emissions_networks[j][k]
            mu_emissions = Data.mean(axis=1)
            axes_other[j-1].plot(property_values_list_col, mu_emissions, label=row_titles[k], c=colors[k])

            mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
            # Plot the 95% confidence interval as a shaded area
            axes_other[j-1].fill_between(property_values_list_col, lower_bound, upper_bound, color=colors[k], alpha=0.3)

    fig_other.supxlabel(r"Carbon tax, $\tau$", fontsize="12")
    fig_other.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    axes_other[1].legend(fontsize="8")

    # Save the Stochastic Block Model and Scale-free networks figure
    plotName_other = fileName + "/Plots"
    f_other = plotName_other + "/sbm_scale_free_tau_emissions_confidence"
    fig_other.savefig(f_other + ".png", dpi=600, format="png")


def plot_means_end_points_emissions_confidence_split_gradient(
    fileName, emissions_networks, property_values_list_col, property_values_list_row, network_titles, row_titles, name
):
    # Create color mapping based on substitutability values
    
    cmap = get_cmap(name)
    
    # Normalize substitutability values to [0,1] for color mapping
    subs_values = np.array([float(title.split('=')[1]) for title in row_titles])
    norm_subs = (subs_values - subs_values.min()) / (subs_values.max() - subs_values.min())
    colors = [cmap(val) for val in norm_subs]

    # First figure: Small-world network
    ncols_sw = 1
    fig_sw, ax_sw = plt.subplots(ncols=ncols_sw, nrows=1, figsize=(6,6), constrained_layout=True)

    ax_sw.set_title(network_titles[0], fontsize="12")
    for k in range(len(property_values_list_row)):
        ax_sw.grid()
        Data = emissions_networks[0][k]
        mu_emissions = Data.mean(axis=1)
        ax_sw.plot(property_values_list_col, mu_emissions, label=row_titles[k], c=colors[k])

        mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
        ax_sw.fill_between(property_values_list_col, lower_bound, upper_bound, color=colors[k], alpha=0.3)

    fig_sw.supxlabel(r"Carbon tax, $\tau$", fontsize="12")
    fig_sw.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    ax_sw.legend(fontsize="8")

    # Save the small-world network figure
    plotName_sw = fileName + "/Plots"
    f_sw = plotName_sw + "/small_world_tau_emissions_confidence"
    fig_sw.savefig(f_sw + ".png", dpi=600, format="png")
    
    # Second figure: Stochastic Block Model and Scale-free networks
    ncols_other = 2
    fig_other, axes_other = plt.subplots(ncols=ncols_other, nrows=1, figsize=(12,6), constrained_layout=True)

    for j in range(1, 3):  # Loop over the other two networks
        axes_other[j-1].set_title(network_titles[j], fontsize="12")
        for k in range(len(property_values_list_row)):
            axes_other[j-1].grid()
            Data = emissions_networks[j][k]
            mu_emissions = Data.mean(axis=1)
            axes_other[j-1].plot(property_values_list_col, mu_emissions, label=row_titles[k], c=colors[k])

            mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
            axes_other[j-1].fill_between(property_values_list_col, lower_bound, upper_bound, color=colors[k], alpha=0.3)

    fig_other.supxlabel(r"Carbon tax, $\tau$", fontsize="12")
    fig_other.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    axes_other[1].legend(fontsize="8")

    # Save the Stochastic Block Model and Scale-free networks figure
    plotName_other = fileName + "/Plots"
    f_other = plotName_other + "/sbm_scale_free_tau_emissions_confidence"
    fig_other.savefig(f_other + ".png", dpi=600, format="png")

def main(
    fileName = "results/tax_sweep_11_29_20__28_09_2023"
) -> None:
    
    name = "plasma"#"magma"#"viridis"#"Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors= cmap.colors  # type: list
    #print(colors)

    #quit()
    emissions_networks = load_object(fileName + "/Data","emissions_data_networks")
    #print(emissions_networks.shape)
    #quit()
    network_titles = ["Small-World", "Stochastic Block Model", "Scale-Free"]
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    base_params = load_object(fileName + "/Data", "base_params") 
    print("base_params",base_params)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]
    #row_label = row_dict["property_title"]#r"Attitude Beta parameters, $(a,b)$"#r"Number of behaviours per agent, M"
    #col_label = col_dict["property_ttitle"]#r'Confirmation bias, $\theta$'
    property_values_list_col = col_dict["property_vals"]
    property_values_list_row = row_dict["property_vals"]

    row_titles = ["Elasticity of substitution, $\sigma$ = %s" % (round(i,3)) for i in property_values_list_row]
    #plot_means_end_points_emissions(fileName, emissions_networks, property_values_list_col, property_values_list_row,network_titles,row_titles,colors)
    #plot_means_end_points_emissions_confidence(fileName, emissions_networks, property_values_list_col, property_values_list_row,network_titles,row_titles,colors)
    #plot_means_end_points_emissions_confidence_split(fileName, emissions_networks, property_values_list_col, property_values_list_row,network_titles,row_titles,colors)
    plot_means_end_points_emissions_confidence_split_gradient(fileName, emissions_networks, property_values_list_col, property_values_list_row,network_titles,row_titles, name)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/network_sub_tau_18_45_43__18_10_2024"
    )