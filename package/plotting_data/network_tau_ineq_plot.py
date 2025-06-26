import matplotlib.pyplot as plt
from package.resources.utility import (
    load_object,
    calc_bounds
)
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_means_end_points_emissions_confidence_split_gradient(
    fileName,
    emissions_networks,
    property_values_list_col,
    property_values_list_row,
    network_titles,
    row_titles,
    name,
    emissions_networks_ref=None  # <-- Add this parameter
):
    # Create color mapping
    cmap = get_cmap(name)
    subs_values = np.array(property_values_list_row)
    norm_subs = (subs_values - subs_values.min()) / (subs_values.max() - subs_values.min())
    colors = [cmap(val) for val in norm_subs]

    # ---------- Small-World Network ----------
    fig_sw, ax_sw = plt.subplots(ncols=1, nrows=1, figsize=(10,6), constrained_layout=True)
    ax_sw.set_title(network_titles[0], fontsize="12")
    
    for k in range(len(property_values_list_row)):
        ax_sw.grid()
        Data = emissions_networks[0][k]
        mu_emissions = Data.mean(axis=1)
        ax_sw.plot(property_values_list_col, mu_emissions, label=row_titles[k], c=colors[k])
        mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
        ax_sw.fill_between(property_values_list_col, lower_bound, upper_bound, color=colors[k], alpha=0.3)

    # Reference line for SW
    if emissions_networks_ref is not None:
        ref_data = emissions_networks_ref[0]
        mu_ref = ref_data.mean(axis=1)
        ax_sw.plot(property_values_list_col, mu_ref, linestyle='--', color='black', label='Reference')

    fig_sw.supxlabel(r"Carbon tax, $\tau$", fontsize="12")
    fig_sw.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    ax_sw.legend(fontsize="8")

    f_sw = fileName + "/Plots/small_world_tau_emissions_confidence"
    fig_sw.savefig(f_sw + ".png", dpi=300, format="png")

    # ---------- SBM & Scale-Free ----------
    fig_other, axes_other = plt.subplots(ncols=2, nrows=1, figsize=(12,6), constrained_layout=True)

    for j in range(1, 3):
        ax = axes_other[j-1]
        ax.set_title(network_titles[j], fontsize="12")
        for k in range(len(property_values_list_row)):
            ax.grid()
            Data = emissions_networks[j][k]
            mu_emissions = Data.mean(axis=1)
            ax.plot(property_values_list_col, mu_emissions, label=row_titles[k], c=colors[k])
            mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
            ax.fill_between(property_values_list_col, lower_bound, upper_bound, color=colors[k], alpha=0.3)

        # Reference line
        if emissions_networks_ref is not None:
            ref_data = emissions_networks_ref[j]
            mu_ref = ref_data.mean(axis=1)
            ax.plot(property_values_list_col, mu_ref, linestyle='--', color='black', label='Reference')

    fig_other.supxlabel(r"Carbon tax, $\tau$", fontsize="12")
    fig_other.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    axes_other[1].legend(fontsize="8")

    f_other = fileName + "/Plots/sbm_scale_free_tau_emissions_confidence"
    fig_other.savefig(f_other + ".png", dpi=300, format="png")

def compute_gini_for_beta(a, b, N=3000, seed=0):
    np.random.seed(seed)
    x = np.random.beta(a, b, size=N)
    x /= np.sum(x)  # Normalize
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def calc_bounds_1d(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))  # standard error of the mean
    z = norm.ppf(0.5 + confidence / 2)
    lower = mean - z * sem
    upper = mean + z * sem
    return mean, lower, upper

import numpy as np

def plot_means_end_points_emissions_confidence_split_gradient_alt(
    fileName,
    emissions_networks,
    property_values_list_col,
    property_values_list_row,
    network_titles,
    row_titles,
    name,
    emissions_networks_ref=None  # <-- Add reference emissions
):
    cmap = get_cmap(name)
    col_values = np.array(property_values_list_col)
    norm_col = (col_values - col_values.min()) / (col_values.max() - col_values.min())
    colors = [cmap(val) for val in norm_col]

    # ---------- Small-World Network ----------
    fig_sw, ax_sw = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax_sw.set_title(network_titles[0], fontsize="12")

    print("Gradients (slopes) for Small-World network:")
    for j, tau_val in enumerate(property_values_list_col):
        mu_emissions = [emissions_networks[0][k][j].mean() for k in range(len(property_values_list_row))]
        lower, upper = [], []
        for k in range(len(property_values_list_row)):
            _, l, u = calc_bounds_1d(emissions_networks[0][k][j], 0.95)
            lower.append(l)
            upper.append(u)

        ax_sw.plot(property_values_list_row, mu_emissions, label=f"Carbon Tax = {tau_val:.2f}", c=colors[j])
        ax_sw.fill_between(property_values_list_row, lower, upper, color=colors[j], alpha=0.3)

        slope, _ = np.polyfit(property_values_list_row, mu_emissions, 1)
        print(f"  τ = {tau_val:.2f} → slope = {slope:.4f}")

        # Reference line for this τ (horizontal)
        if emissions_networks_ref is not None:
            ref_val = emissions_networks_ref[0][j].mean()
            ax_sw.plot(property_values_list_row, [ref_val]*len(property_values_list_row),
                       linestyle='--', color='black' if j == 0 else 'gray', alpha=0.6)

    ax_sw.set_xlabel(r"a in Beta distribution", fontsize="12")
    ax_sw.set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    ax_sw.legend(fontsize="8")
    f_sw = f"{fileName}/Plots/small_world_a_emissions_confidence_alt"
    fig_sw.savefig(f_sw + ".png", dpi=300, format="png")

    # ---------- SBM & Scale-Free ----------
    fig_other, axes_other = plt.subplots(ncols=2, figsize=(12, 6), constrained_layout=True)
    for i, ax in enumerate(axes_other):
        ax.set_title(network_titles[i + 1], fontsize="12")
        print(f"\nGradients (slopes) for {network_titles[i + 1]} network:")
        for j, tau_val in enumerate(property_values_list_col):
            mu_emissions = [emissions_networks[i + 1][k][j].mean() for k in range(len(property_values_list_row))]
            lower, upper = [], []
            for k in range(len(property_values_list_row)):
                _, l, u = calc_bounds_1d(emissions_networks[i + 1][k][j], 0.95)
                lower.append(l)
                upper.append(u)

            ax.plot(property_values_list_row, mu_emissions, label=f"Carbon Tax = {tau_val:.2f}", c=colors[j])
            ax.fill_between(property_values_list_row, lower, upper, color=colors[j], alpha=0.3)

            slope, _ = np.polyfit(property_values_list_row, mu_emissions, 1)
            print(f"  τ = {tau_val:.2f} → slope = {slope:.4f}")

            # Reference line
            if emissions_networks_ref is not None:
                ref_val = emissions_networks_ref[i + 1][j].mean()
                ax.plot(property_values_list_row, [ref_val]*len(property_values_list_row),
                        linestyle='--', color='black' if j == 0 else 'gray', alpha=0.6)

    fig_other.supxlabel(r"a in Beta distribution", fontsize="12")
    fig_other.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    axes_other[1].legend(fontsize="8")
    f_other = f"{fileName}/Plots/sbm_scale_free_a_emissions_confidence_alt"
    fig_other.savefig(f_other + ".png", dpi=300, format="png")

def plot_emissions_vs_gini_scatter(
    fileName,
    emissions_networks,
    gini_networks,
    property_values_list_col,
    property_values_list_row,
    network_titles,
    name,
    emissions_networks_ref=None,
    gini_networks_ref=None
):
    from matplotlib.cm import get_cmap
    import numpy as np

    cmap = get_cmap(name)
    col_values = np.array(property_values_list_col)
    norm_col = (col_values - col_values.min()) / (col_values.max() - col_values.min())
    colors = [cmap(val) for val in norm_col]

    def extract_mean_emissions_and_gini(network_index, tau_index):
        x_gini = []
        y_emissions = []
        y_lower = []
        y_upper = []
        for k in range(len(property_values_list_row)):
            gini_mean, _, _ = calc_bounds_1d(gini_networks[network_index][k][0])
            em_data = emissions_networks[network_index][k][tau_index]
            mu, l, u = calc_bounds_1d(em_data)
            x_gini.append(gini_mean)
            y_emissions.append(mu)
            y_lower.append(l)
            y_upper.append(u)
        return np.array(x_gini), np.array(y_emissions), np.array(y_lower), np.array(y_upper)

    def extract_reference_points(network_index):
        gini_means = []
        emissions_means = []
        for tau_index in range(len(property_values_list_col)):
            gini_mean, _, _ = calc_bounds_1d(gini_networks_ref[network_index][tau_index])
            em_mean, _, _ = calc_bounds_1d(emissions_networks_ref[network_index][tau_index])
            gini_means.append(gini_mean)
            emissions_means.append(em_mean)
        return np.array(gini_means), np.array(emissions_means)

    # === SMALL-WORLD ===
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title(f"{network_titles[0]}", fontsize="12")

    for j, tau_val in enumerate(property_values_list_col):
        x_gini, y_emissions, y_lower, y_upper = extract_mean_emissions_and_gini(0, j)
        ax.scatter(x_gini, y_emissions, color=colors[j], alpha=0.8)
        ax.errorbar(x_gini, y_emissions, yerr=[y_emissions - y_lower, y_upper - y_emissions],
                    fmt='none', color=colors[j], alpha=0.3)

        slope, intercept = np.polyfit(x_gini, y_emissions, 1)
        x_fit = np.linspace(min(x_gini), max(x_gini), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color=colors[j], linestyle='--',
                label=f"τ = {tau_val:.2f} (slope = {slope:.2f})")

    # Reference line
    if emissions_networks_ref is not None and gini_networks_ref is not None:
        x_ref, y_ref = extract_reference_points(0)
        ax.scatter(x_ref, y_ref, color='black', marker='o', label='Reference')

    ax.set_xlabel("Gini coefficient", fontsize=12)
    ax.set_ylabel("Cumulative carbon emissions, E", fontsize=12)
    ax.legend(fontsize=8)
    fig.savefig(f"{fileName}/Plots/small_world_emissions_vs_gini.png", dpi=300)

    # === SBM and SCALE-FREE ===
    fig_other, axes = plt.subplots(ncols=2, figsize=(12, 6), constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.set_title(f"{network_titles[i+1]}", fontsize="12")
        for j, tau_val in enumerate(property_values_list_col):
            x_gini, y_emissions, y_lower, y_upper = extract_mean_emissions_and_gini(i+1, j)

            ax.scatter(x_gini, y_emissions, color=colors[j], alpha=0.8)
            ax.errorbar(x_gini, y_emissions, yerr=[y_emissions - y_lower, y_upper - y_emissions],
                        fmt='none', color=colors[j], alpha=0.3)

            slope, intercept = np.polyfit(x_gini, y_emissions, 1)
            x_fit = np.linspace(min(x_gini), max(x_gini), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color=colors[j], linestyle='--',
                    label=f"τ = {tau_val:.2f} (slope = {slope:.2f})")

        # Reference line
        if emissions_networks_ref is not None and gini_networks_ref is not None:
            x_ref, y_ref = extract_reference_points(i+1)
            ax.scatter(x_ref, y_ref, color='black', marker='o', label='Reference')

        ax.set_xlabel("Gini coefficient", fontsize=12)
        ax.set_ylabel("Cumulative carbon emissions, E", fontsize=12)

    axes[1].legend(fontsize=8)
    fig_other.savefig(f"{fileName}/Plots/sbm_scale_free_emissions_vs_gini.png", dpi=300)

def plot_emissions_vs_gini_scatter_alt(
    fileName,
    emissions_networks,
    gini_networks,
    property_values_list_col,
    property_values_list_row,
    network_titles,
    name,
    emissions_networks_ref=None,
    gini_networks_ref=None
):
    from matplotlib.cm import get_cmap
    import numpy as np

    cmap = get_cmap(name)
    col_values = np.array(property_values_list_col)
    norm_col = (col_values - col_values.min()) / (col_values.max() - col_values.min())
    colors = [cmap(val) for val in norm_col]

    def extract_mean_emissions_and_gini(network_index, tau_index):
        x_gini, y_emissions, y_lower, y_upper = [], [], [], []
        for k in range(len(property_values_list_row)):
            gini_mean, _, _ = calc_bounds_1d(gini_networks[network_index][k][0])
            em_data = emissions_networks[network_index][k][tau_index]
            mu, l, u = calc_bounds_1d(em_data)
            x_gini.append(gini_mean)
            y_emissions.append(mu)
            y_lower.append(l)
            y_upper.append(u)
        return np.array(x_gini), np.array(y_emissions), np.array(y_lower), np.array(y_upper)

    def extract_reference_emissions(network_index):
        emissions_means = []
        for tau_index in range(len(property_values_list_col)):
            em_mean, _, _ = calc_bounds_1d(emissions_networks_ref[network_index][tau_index])
            emissions_means.append(em_mean)
        return np.array(emissions_means)

    # === SMALL-WORLD ===
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title(f"{network_titles[0]}", fontsize="12")

    for j, tau_val in enumerate(property_values_list_col):
        x_gini, y_emissions, y_lower, y_upper = extract_mean_emissions_and_gini(0, j)

        ax.scatter(x_gini, y_emissions, color=colors[j], alpha=0.8)
        ax.errorbar(x_gini, y_emissions, yerr=[y_emissions - y_lower, y_upper - y_emissions],
                    fmt='none', color=colors[j], alpha=0.3)

        slope, intercept = np.polyfit(x_gini, y_emissions, 1)
        x_fit = np.linspace(min(x_gini), max(x_gini), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color=colors[j], linestyle='--',
                label=f"τ = {tau_val:.2f} (slope = {slope:.2f})")

    # Add reference horizontal lines (color-matched)
    if emissions_networks_ref is not None:
        y_refs = extract_reference_emissions(0)
        for j, y_val in enumerate(y_refs):
            ax.axhline(y=y_val, linestyle='-.', color="black", alpha=0.6, label='Reference')

    ax.set_xlabel("Gini coefficient", fontsize=12)
    ax.set_ylabel("Cumulative carbon emissions, E", fontsize=12)
    ax.legend(fontsize=8)
    fig.savefig(f"{fileName}/Plots/small_world_emissions_vs_gini_alt.png", dpi=300)

    # === SBM and SCALE-FREE ===
    fig_other, axes = plt.subplots(ncols=2, figsize=(12, 6), constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.set_title(f"{network_titles[i+1]}", fontsize="12")
        for j, tau_val in enumerate(property_values_list_col):
            x_gini, y_emissions, y_lower, y_upper = extract_mean_emissions_and_gini(i+1, j)

            ax.scatter(x_gini, y_emissions, color=colors[j], alpha=0.8)
            ax.errorbar(x_gini, y_emissions, yerr=[y_emissions - y_lower, y_upper - y_emissions],
                        fmt='none', color=colors[j], alpha=0.3)

            slope, intercept = np.polyfit(x_gini, y_emissions, 1)
            x_fit = np.linspace(min(x_gini), max(x_gini), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color=colors[j], linestyle='--',
                    label=f"τ = {tau_val:.2f} (slope = {slope:.2f})")

        # Reference horizontal lines
        if emissions_networks_ref is not None:
            y_refs = extract_reference_emissions(i + 1)
            for j, y_val in enumerate(y_refs):
                ax.axhline(y=y_val, linestyle='-.', color="black", alpha=0.6, label='Reference')

        ax.set_xlabel("Gini coefficient", fontsize=12)
        ax.set_ylabel("Cumulative carbon emissions, E", fontsize=12)

    axes[1].legend(fontsize=8)
    fig_other.savefig(f"{fileName}/Plots/sbm_scale_free_emissions_vs_gini_alt.png", dpi=300)

def plot_emissions_vs_gini_scatter_broken(
    fileName,
    emissions_networks,
    gini_networks,
    property_values_list_col,
    property_values_list_row,
    network_titles,
    name,
    emissions_networks_ref=None,
    gini_networks_ref=None
):
    from matplotlib.cm import get_cmap
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    cmap = get_cmap(name)
    norm_col = (np.array(property_values_list_col) - min(property_values_list_col)) / (
        max(property_values_list_col) - min(property_values_list_col)
    )
    colors = [cmap(val) for val in norm_col]

    def extract_mean_emissions_and_gini(network_index, tau_index):
        x_gini, y_emissions, y_lower, y_upper = [], [], [], []
        for k in range(len(property_values_list_row)):
            gini_mean, _, _ = calc_bounds_1d(gini_networks[network_index][k][0])
            em_data = emissions_networks[network_index][k][tau_index]
            mu, l, u = calc_bounds_1d(em_data)
            x_gini.append(gini_mean)
            y_emissions.append(mu)
            y_lower.append(l)
            y_upper.append(u)
        return np.array(x_gini), np.array(y_emissions), np.array(y_lower), np.array(y_upper)

    def extract_reference_emissions(network_index):
        emissions_means = []
        for tau_index in range(len(property_values_list_col)):
            em_mean, _, _ = calc_bounds_1d(emissions_networks_ref[network_index][tau_index])
            emissions_means.append(em_mean)
        return np.array(emissions_means)

    def plot_broken_axes_subplot(ax1, ax2, network_index):
        if emissions_networks_ref is not None:
            y_refs = extract_reference_emissions(network_index)
            for j, y_val in enumerate(y_refs):
                ax1.scatter(
                    0, y_val,
                    color=colors[j],
                    marker='X',
                    edgecolor='black',
                    s=70,
                    zorder=5
                )

        min_x = float("inf")
        for j, tau_val in enumerate(property_values_list_col):
            x_gini, y_emissions, y_lower, y_upper = extract_mean_emissions_and_gini(network_index, j)
            min_x = min(min_x, np.min(x_gini))

            ax2.scatter(x_gini, y_emissions, color=colors[j], alpha=0.8)
            ax2.errorbar(
                x_gini, y_emissions,
                yerr=[y_emissions - y_lower, y_upper - y_emissions],
                fmt='none', color=colors[j], alpha=0.3
            )

            slope, intercept = np.polyfit(x_gini, y_emissions, 1)
            x_fit = np.linspace(min(x_gini), max(x_gini), 100)
            y_fit = slope * x_fit + intercept
            ax2.plot(x_fit, y_fit, color=colors[j], linestyle='--',
                     label=f"Carbon tax, τ = {tau_val:.2f} (slope = {slope:.2f})")

        ax1.set_xlim(-0.05, 0.05)
        ax2.set_xlim(min_x - 0.01, None)
        ax1.set_xticks([0])

        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_left()
        ax1.grid()
        ax2.grid()
        ax1.set_ylim(ax2.get_ylim())

        # Diagonal break marks
        d = .015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot([1 - d, 1 + d], [-d, +d], **kwargs)
        ax1.plot([1 - d, 1 + d], [1 - d, 1 + d], **kwargs)

        kwargs.update(transform=ax2.transAxes)
        ax2.plot([-d, +d], [-d, +d], **kwargs)
        ax2.plot([-d, +d], [1 - d, 1 + d], **kwargs)

        # Legend on ax2
        ref_handle = mlines.Line2D([], [], color='gray', marker='X', linestyle='None',
                                   markeredgecolor='black', markersize=8, label="Equal expenditure reference")
        handles, labels = ax2.get_legend_handles_labels()
        handles.append(ref_handle)
        labels.append("Equal expenditure reference")
        ax2.legend(handles, labels, fontsize=8)

    # === FIGURE 1: Small-World ===
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6), sharey=True,
        gridspec_kw={"width_ratios": [0.5, 5]}, constrained_layout=True
    )
    fig.suptitle(network_titles[0], fontsize=12)
    plot_broken_axes_subplot(ax1, ax2, 0)
    ax2.set_xlabel("Gini coefficient", fontsize=12)
    ax1.set_ylabel("Cumulative carbon emissions, E", fontsize=12)
    fig.savefig(f"{fileName}/Plots/small_world_emissions_vs_gini_broken_axis.png", dpi=300)

    # === FIGURE 2: SBM + SF side-by-side ===
    fig, axes = plt.subplots(
        1, 4, figsize=(16, 6), sharey=True,
        gridspec_kw={"width_ratios": [0.5, 5, 0.5, 5]}, constrained_layout=True
    )
    fig.suptitle("Stochastic Block Model and Scale-Free", fontsize=12)

    # SBM (network index 1)
    plot_broken_axes_subplot(axes[0], axes[1], 1)
    axes[1].set_xlabel("Gini coefficient", fontsize=12)
    axes[0].set_ylabel("Cumulative carbon emissions, E", fontsize=12)

    # SF (network index 2)
    plot_broken_axes_subplot(axes[2], axes[3], 2)
    axes[3].set_xlabel("Gini coefficient", fontsize=12)

    fig.savefig(f"{fileName}/Plots/sbm_scale_free_emissions_vs_gini_broken_axis.png", dpi=300)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.lines as mlines

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
    fileName = "results/network_ineq_tau_12_00_32__11_06_2025"
) -> None:
    # Load main simulation data
    emissions_networks = load_object(fileName + "/Data", "emissions_data_networks")
    gini_networks = load_object(fileName + "/Data", "gini_array")
    poorest_networks = load_object(fileName + "/Data", "poorest_spend_prop_array")
    richest_networks = load_object(fileName + "/Data", "richest_spend_prop_array")
    
    # Load reference data
    emissions_networks_ref = load_object(fileName + "/Data", "emissions_data_networks_ref")
    gini_networks_ref = load_object(fileName + "/Data", "gini_array_ref")

    network_titles = ["Small-World", "Stochastic Block Model", "Scale-Free"]
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]
    property_values_list_col = col_dict["property_vals"]
    property_values_list_row = row_dict["property_vals"]

    base_params = load_object(fileName + "/Data", "base_params")
    b_expenditure = base_params["b_expenditure"]

    # Create row titles with statistical summaries
    row_titles = []
    for i, a in enumerate(property_values_list_row):
        gini_samples = gini_networks[0][i][0]
        poorest_samples = poorest_networks[0][i][0]
        richest_samples = richest_networks[0][i][0]
        mean_gini, lower_gini, upper_gini = calc_bounds_1d(gini_samples, 0.95)
        mean_poorest, _, _ = calc_bounds_1d(poorest_samples, 0.95)
        mean_richest, _, _ = calc_bounds_1d(richest_samples, 0.95)
        row_titles.append(
            f"a Beta distribution, Expenditure = {np.round(a, 5)}, "
            f"Gini = {np.round(mean_gini, 5)}, "
            f"Poorest Prop = {np.round(mean_poorest, 5)}, "
            f"Richest Prop = {np.round(mean_richest, 5)}"
        )

    name = "plasma"
    """
    # Plotting with reference lines
    plot_means_end_points_emissions_confidence_split_gradient(
        fileName,
        emissions_networks,
        property_values_list_col,
        property_values_list_row,
        network_titles,
        row_titles,
        name,
        emissions_networks_ref=emissions_networks_ref
    )

    plot_means_end_points_emissions_confidence_split_gradient_alt(
        fileName,
        emissions_networks,
        property_values_list_col,
        property_values_list_row,
        network_titles,
        row_titles,
        name,
        emissions_networks_ref=emissions_networks_ref
    )

    plot_emissions_vs_gini_scatter(
        fileName,
        emissions_networks,
        gini_networks,
        property_values_list_col,
        property_values_list_row,
        network_titles,
        name,
        emissions_networks_ref=emissions_networks_ref,
        gini_networks_ref=gini_networks_ref
    )


    plot_emissions_vs_gini_scatter_broken(
        fileName,
        emissions_networks,
        gini_networks,
        property_values_list_col,
        property_values_list_row,
        network_titles,
        name,
        emissions_networks_ref=emissions_networks_ref,
        gini_networks_ref=gini_networks_ref
    )
    """

    fileName_without = "results/network_ineq_tau_20_10_24__25_06_2025"
    # Load WITHOUT REDISTRIBUTION main simulation data
    emissions_networks_without = load_object(fileName_without + "/Data", "emissions_data_networks")
    gini_networks_without = load_object(fileName_without + "/Data", "gini_array")
    emissions_networks_ref_without = load_object(fileName_without + "/Data", "emissions_data_networks_ref")

    data_dict = {
        "with_dividend": {
            "emissions": emissions_networks,
            "gini": gini_networks,
            "emissions_ref": emissions_networks_ref
        },
        "no_dividend": {
            "emissions": emissions_networks_without,
            "gini": gini_networks_without,
            "emissions_ref": emissions_networks_ref_without
        }
    }

    plot_emissions_vs_gini_dual_broken(
        fileName=fileName,
        data_dict=data_dict,
        property_values_list_col=property_values_list_col,
        property_values_list_row=property_values_list_row,
        network_titles=network_titles,
        name=name
    )


    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/network_ineq_tau_13_42_03__26_06_2025"#network_ineq_tau_00_14_13__18_06_2025"#network_ineq_tau_17_24_33__17_06_2025"#network_ineq_tau_11_50_31__17_06_2025"#network_ineq_tau_10_27_38__17_06_2025"#network_ineq_tau_11_59_40__11_06_2025"
    )