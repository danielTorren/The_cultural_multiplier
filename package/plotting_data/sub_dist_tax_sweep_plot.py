import matplotlib.pyplot as plt
from package.resources.utility import (
    load_object,
    calc_bounds
)
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from package.resources.utility import calc_bounds

def plot_means_end_points_emissions_confidence_split_gradient(
    fileName,
    emissions_networks,
    property_values_list_col,
    property_values_list_row,
    network_titles,
    row_titles,
    name,
    emissions_networks_ref=None  # Reference case added here
):
    # Create color mapping based on substitutability values
    cmap = get_cmap(name)
    
    # Normalize substitutability values to [0,1] for color mapping
    subs_values = np.array([float(title.split('=')[1]) for title in row_titles])
    norm_subs = (subs_values - subs_values.min()) / (subs_values.max() - subs_values.min())
    colors = [cmap(val) for val in norm_subs]

    # ---------- Small-World Network Plot ----------
    fig_sw, ax_sw = plt.subplots(ncols=1, nrows=1, figsize=(6,6), constrained_layout=True)

    ax_sw.set_title(network_titles[0], fontsize="12")
    for k in range(len(property_values_list_row)):
        ax_sw.grid()
        Data = emissions_networks[0][k]
        mu_emissions = Data.mean(axis=1)
        ax_sw.plot(property_values_list_col, mu_emissions, label=row_titles[k], c=colors[k])

        mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
        ax_sw.fill_between(property_values_list_col, lower_bound, upper_bound, color=colors[k], alpha=0.3)

    # Plot reference line for Small-World
    if emissions_networks_ref is not None:
        ref_data = emissions_networks_ref[0]
        mu_ref = ref_data.mean(axis=1)
        ax_sw.plot(property_values_list_col, mu_ref, linestyle='--', color='black', label='Reference')

    fig_sw.supxlabel(r"Carbon tax, $\tau$", fontsize="12")
    fig_sw.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    ax_sw.legend(fontsize="8")

    # Save Small-World plot
    plotName_sw = fileName + "/Plots"
    f_sw = plotName_sw + "/small_world_tau_emissions_confidence"
    fig_sw.savefig(f_sw + ".png", dpi=300, format="png")

    # ---------- SBM and Scale-Free Network Plot ----------
    fig_other, axes_other = plt.subplots(ncols=2, nrows=1, figsize=(12,6), constrained_layout=True)

    for j in range(1, 3):  # SBM and SF
        axes_other[j-1].set_title(network_titles[j], fontsize="12")
        for k in range(len(property_values_list_row)):
            axes_other[j-1].grid()
            Data = emissions_networks[j][k]
            mu_emissions = Data.mean(axis=1)
            axes_other[j-1].plot(property_values_list_col, mu_emissions, label=row_titles[k], c=colors[k])

            mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
            axes_other[j-1].fill_between(property_values_list_col, lower_bound, upper_bound, color=colors[k], alpha=0.3)

        # Plot reference line
        if emissions_networks_ref is not None:
            ref_data = emissions_networks_ref[j]
            mu_ref = ref_data.mean(axis=1)
            axes_other[j-1].plot(property_values_list_col, mu_ref, linestyle='--', color='black', label='Reference')

    fig_other.supxlabel(r"Carbon tax, $\tau$", fontsize="12")
    fig_other.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    axes_other[1].legend(fontsize="8")

    # Save SBM and SF plot
    plotName_other = fileName + "/Plots"
    f_other = plotName_other + "/sbm_scale_free_tau_emissions_confidence"
    fig_other.savefig(f_other + ".png", dpi=300, format="png")

def main(
    fileName = "results/tax_sweep_11_29_20__28_09_2023"
) -> None:

    emissions_networks = load_object(fileName + "/Data","emissions_data_networks")
    network_titles = ["Small-World", "Stochastic Block Model", "Scale-Free"]
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    emissions_networks_ref = load_object(fileName + "/Data","emissions_data_networks_ref")

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]
    property_values_list_col = col_dict["property_vals"]
    property_values_list_row = row_dict["property_vals"]

    #row_titles = ["Mean elasticity of substitution = %s" % (round(i,3)) for i in property_values_list_row]
    row_titles = ["a Beta distribution, Elasticity of substitution = %s" % (round(i,3)) for i in property_values_list_row]
    name = "plasma"

    plot_means_end_points_emissions_confidence_split_gradient(
        fileName, emissions_networks, property_values_list_col, property_values_list_row,
        network_titles, row_titles, name, emissions_networks_ref=emissions_networks_ref
    )
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/sub_dist_tax_sweep_19_19_41__07_07_2025"#sub_dist_tax_sweep_11_43_29__16_06_2025"
    )