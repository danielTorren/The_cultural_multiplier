import matplotlib.pyplot as plt
from package.resources.utility import (
    load_object
)
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.cm import get_cmap
import matplotlib.lines as mlines

def calc_bounds_1d(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))  # standard error of the mean
    z = norm.ppf(0.5 + confidence / 2)
    lower = mean - z * sem
    upper = mean + z * sem
    return mean, lower, upper


def plot_emissions_vs_gini_dual_broken(
    fileName,
    data_dict,  # A dictionary with "with_dividend" and "no_dividend" keys
    property_values_list_col,
    property_values_list_row,
    network_titles,
    name
):
    cmap = get_cmap(name)
    norm_col = (np.array(property_values_list_col) - min(property_values_list_col)) / (
        max(property_values_list_col) - min(property_values_list_col)
    )
    colors = [cmap(val) for val in norm_col]

    def extract(network_data, index, tau_index):
        gini_arr = network_data["gini"]
        emissions_arr = network_data["emissions"]
        x_gini, y_emissions, y_lower, y_upper = [], [], [], []
        for k in range(len(property_values_list_row)):
            gini_mean, _, _ = calc_bounds_1d(gini_arr[index][k][0])
            em_data = emissions_arr[index][k][tau_index]
            mu, l, u = calc_bounds_1d(em_data)
            x_gini.append(gini_mean)
            y_emissions.append(mu)
            y_lower.append(l)
            y_upper.append(u)
        return np.array(x_gini), np.array(y_emissions), np.array(y_lower), np.array(y_upper)

    def extract_refs(emissions_ref, index):
        refs = []
        for tau_index in range(len(property_values_list_col)):
            mu, _, _ = calc_bounds_1d(emissions_ref[index][tau_index])
            refs.append(mu)
        return np.array(refs)

    def plot_subplot(ax1, ax2, net_idx, data_key, label_prefix, linestyle, with_state):
        data = data_dict[data_key]
        emissions_ref = data.get("emissions_ref")

        if emissions_ref is not None:
            y_refs = extract_refs(emissions_ref, net_idx)
            for j, y_val in enumerate(y_refs):
                marker = 'P' if with_state else 'X'
                ax1.scatter(0, y_val, color=colors[j], marker=marker, edgecolor='black', s=70, zorder=5)

        min_x = float("inf")
        for j, tau_val in enumerate(property_values_list_col):
            x_gini, y_em, y_l, y_u = extract(data, net_idx, j)
            min_x = min(min_x, np.min(x_gini))

            ax2.scatter(x_gini, y_em, color=colors[j], alpha=0.8, label=None)
            ax2.errorbar(x_gini, y_em, yerr=[y_em - y_l, y_u - y_em],
                         fmt='none', color=colors[j], alpha=0.3)

            slope, intercept = np.polyfit(x_gini, y_em, 1)
            x_fit = np.linspace(min(x_gini), max(x_gini), 100)
            y_fit = slope * x_fit + intercept
            ax2.plot(x_fit, y_fit, color=colors[j], linestyle=linestyle,
                     label=f"{label_prefix} τ={tau_val:.2f} (slope={slope:.2f})")

        ax1.set_xlim(-0.05, 0.05)
        ax2.set_xlim(min_x - 0.01, None)
        ax1.set_xticks([0])

        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_left()

        ax1.grid(True)
        ax2.grid(True)

        ax1.set_ylim(ax2.get_ylim())

        # Diagonal break marks
        d = .015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot([1 - d, 1 + d], [-d, +d], **kwargs)
        ax1.plot([1 - d, 1 + d], [1 - d, 1 + d], **kwargs)

        kwargs.update(transform=ax2.transAxes)
        ax2.plot([-d, +d], [-d, +d], **kwargs)
        ax2.plot([-d, +d], [1 - d, 1 + d], **kwargs)

    # === Small-World Plot ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True,
                                   gridspec_kw={"width_ratios": [0.5, 5]}, constrained_layout=True)
    fig.suptitle(network_titles[0], fontsize=12)

    plot_subplot(ax1, ax2, 0, "with_dividend", "With carbon dividend,", "--", True)
    plot_subplot(ax1, ax2, 0, "no_dividend", "No carbon dividend,", ":", False)

    ax2.set_xlabel("Gini coefficient", fontsize=12)
    ax1.set_ylabel("Cumulative carbon emissions, E", fontsize=12)

    # Create and sort legend
    handles, labels = ax2.get_legend_handles_labels()

    ref_with = mlines.Line2D([], [], color='gray', marker='P', linestyle='None',
                             markeredgecolor='black', markersize=8, label="With carbon dividend, Equal expenditure reference")
    ref_no = mlines.Line2D([], [], color='gray', marker='X', linestyle='None',
                           markeredgecolor='black', markersize=8, label="No carbon dividend, Equal expenditure reference")

    handles.extend([ref_with, ref_no])
    labels.extend([ref_with.get_label(), ref_no.get_label()])

    # Sort: "with" first, then "no"
    legend_items = [(h, l) for h, l in zip(handles, labels)]
    with_items = [item for item in legend_items if item[1].startswith("With")]
    no_items = [item for item in legend_items if item[1].startswith("No")]
    sorted_handles, sorted_labels = zip(*(with_items + no_items))

    ax2.legend(sorted_handles, sorted_labels, fontsize=8, ncol=2)
    fig.savefig(f"{fileName}/Plots/small_world_emissions_vs_gini_dual.png", dpi=300)

    # === SBM and Scale-Free Combined Plot ===
    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True,
                             gridspec_kw={"width_ratios": [0.5, 5, 0.5, 5]}, constrained_layout=True)
    fig.suptitle("SBM and Scale-Free: With vs. No Carbon Dividend", fontsize=12)

    # SBM (index 1)
    plot_subplot(axes[0], axes[1], 1, "with_dividend", "With carbon dividend,", "--", True)
    plot_subplot(axes[0], axes[1], 1, "no_dividend", "No carbon dividend,", ":", False)
    axes[1].set_xlabel("Gini coefficient", fontsize=12)
    axes[0].set_ylabel("Cumulative carbon emissions, E", fontsize=12)

    # Scale-Free (index 2)
    plot_subplot(axes[2], axes[3], 2, "with_dividend", "With carbon dividend,", "--", True)
    plot_subplot(axes[2], axes[3], 2, "no_dividend", "No carbon dividend,", ":", False)
    axes[3].set_xlabel("Gini coefficient", fontsize=12)

    handles, labels = axes[3].get_legend_handles_labels()
    handles.extend([ref_with, ref_no])
    labels.extend([ref_with.get_label(), ref_no.get_label()])

    legend_items = [(h, l) for h, l in zip(handles, labels)]
    with_items = [item for item in legend_items if item[1].startswith("With")]
    no_items = [item for item in legend_items if item[1].startswith("No")]
    sorted_handles, sorted_labels = zip(*(with_items + no_items))

    axes[3].legend(sorted_handles, sorted_labels, fontsize=8, ncol=2)
    fig.savefig(f"{fileName}/Plots/sbm_scale_free_emissions_vs_gini_dual.png", dpi=300)

def main(
    fileName = "results/network_ineq_tau_13_42_03__26_06_2025"
) -> None:

    # Load data WITH redistribution (carbon dividend)
    emissions_networks_with = load_object(fileName + "/Data", "emissions_data_networks_with_re")
    gini_networks_with = load_object(fileName + "/Data", "gini_array_with_re")
    emissions_networks_ref_with = load_object(fileName + "/Data", "emissions_data_networks_ref_with_re")

    # Load data WITHOUT redistribution
    emissions_networks_without = load_object(fileName + "/Data", "emissions_data_networks")
    gini_networks_without = load_object(fileName + "/Data", "gini_array")
    emissions_networks_ref_without = load_object(fileName + "/Data", "emissions_data_networks_ref")

    # Load variable params and titles
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    property_values_list_col = variable_parameters_dict["col"]["property_vals"]
    property_values_list_row = variable_parameters_dict["row"]["property_vals"]

    network_titles = ["Small-World", "Stochastic Block Model", "Scale-Free"]

    # Structure the data
    data_dict = {
        "with_dividend": {
            "emissions": emissions_networks_with,
            "gini": gini_networks_with,
            "emissions_ref": emissions_networks_ref_with
        },
        "no_dividend": {
            "emissions": emissions_networks_without,
            "gini": gini_networks_without,
            "emissions_ref": emissions_networks_ref_without
        }
    }

    # Plotting
    plot_emissions_vs_gini_dual_broken(
        fileName=fileName,
        data_dict=data_dict,
        property_values_list_col=property_values_list_col,
        property_values_list_row=property_values_list_row,
        network_titles=network_titles,
        name="plasma"
    )

    plt.show()

if __name__ == '__main__':
    main(
        fileName="results/network_ineq_tau_13_42_03__26_06_2025"
    )
