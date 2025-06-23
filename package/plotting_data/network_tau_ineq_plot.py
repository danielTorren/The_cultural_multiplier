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
    fileName, emissions_networks, property_values_list_col, property_values_list_row, network_titles, row_titles, name
):
    # Create color mapping based on substitutability values
    
    cmap = get_cmap(name)
    
    # Normalize the a values (or whatever property varies across rows) directly
    subs_values = np.array(property_values_list_row)
    norm_subs = (subs_values - subs_values.min()) / (subs_values.max() - subs_values.min())
    colors = [cmap(val) for val in norm_subs]


    # First figure: Small-world network
    ncols_sw = 1
    fig_sw, ax_sw = plt.subplots(ncols=ncols_sw, nrows=1, figsize=(10,6), constrained_layout=True)

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
    fig_sw.savefig(f_sw + ".png", dpi=300, format="png")
    
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
    fileName, emissions_networks, property_values_list_col, property_values_list_row, network_titles, row_titles, name
):
    cmap = get_cmap(name)
    col_values = np.array(property_values_list_col)
    norm_col = (col_values - col_values.min()) / (col_values.max() - col_values.min())
    colors = [cmap(val) for val in norm_col]

    # First figure: Small-world network
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

        # Plot
        ax_sw.plot(property_values_list_row, mu_emissions, label=f"Carbon Tax = {tau_val:.2f}", c=colors[j])
        ax_sw.fill_between(property_values_list_row, lower, upper, color=colors[j], alpha=0.3)

        # Fit a line and print the slope
        slope, intercept = np.polyfit(property_values_list_row, mu_emissions, 1)
        print(f"  τ = {tau_val:.2f} → slope = {slope:.4f}")

    fig_sw.supxlabel(r"a in Beta distribution", fontsize="12")
    fig_sw.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    ax_sw.legend(fontsize="8")
    f_sw = f"{fileName}/Plots/small_world_a_emissions_confidence_alt"
    fig_sw.savefig(f_sw + ".png", dpi=300, format="png")

    # SBM and Scale-Free
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

            # Fit a line and print the slope
            slope, intercept = np.polyfit(property_values_list_row, mu_emissions, 1)
            print(f"  τ = {tau_val:.2f} → slope = {slope:.4f}")

    fig_other.supxlabel(r"a in Beta distribution", fontsize="12")
    fig_other.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    axes_other[1].legend(fontsize="8")
    f_other = f"{fileName}/Plots/sbm_scale_free_a_emissions_confidence_alt"
    fig_other.savefig(f_other + ".png", dpi=300, format="png")


def plot_emissions_vs_gini_scatter(
    fileName, emissions_networks, gini_networks, property_values_list_col, property_values_list_row,
    network_titles, name
):
    import matplotlib.pyplot as plt
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

    # === SMALL-WORLD ===
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title(f"{network_titles[0]}", fontsize="12")

    for j, tau_val in enumerate(property_values_list_col):
        x_gini, y_emissions, y_lower, y_upper = extract_mean_emissions_and_gini(0, j)

        # Scatter + error bars
        ax.scatter(x_gini, y_emissions, color=colors[j], alpha=0.8)
        ax.errorbar(x_gini, y_emissions, yerr=[y_emissions - y_lower, y_upper - y_emissions],
                    fmt='none', color=colors[j], alpha=0.3)

        # Fit and plot line
        slope, intercept = np.polyfit(x_gini, y_emissions, 1)
        x_fit = np.linspace(min(x_gini), max(x_gini), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color=colors[j], linestyle='--',
                label=f"τ = {tau_val:.2f} (slope = {slope:.2f})")

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

        ax.set_xlabel("Gini coefficient", fontsize=12)
        ax.set_ylabel("Cumulative carbon emissions, E", fontsize=12)

    axes[1].legend(fontsize=8)
    fig_other.savefig(f"{fileName}/Plots/sbm_scale_free_emissions_vs_gini.png", dpi=300)


def main(
    fileName = "results/network_ineq_tau_12_00_32__11_06_2025"
) -> None:

    emissions_networks = load_object(fileName + "/Data","emissions_data_networks")
    gini_networks = load_object(fileName + "/Data","gini_array")
    poorest_networks = load_object(fileName + "/Data","poorest_spend_prop_array")
    richest_networks = load_object(fileName + "/Data","richest_spend_prop_array")
    network_titles = ["Small-World", "Stochastic Block Model", "Scale-Free"]
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]
    property_values_list_col = col_dict["property_vals"]
    property_values_list_row = row_dict["property_vals"]

    row_titles = ["a Beta distribution, Expenditure = %s" % (round(i,3)) for i in property_values_list_row]
    name = "plasma"

    base_params = load_object(fileName + "/Data", "base_params")
    b_expenditure = base_params["b_expenditure"]
    N = base_params["b_expenditure"]

    # Compute average Gini and 95% confidence interval for each a value
    row_titles = []
    for i, a in enumerate(property_values_list_row):
        gini_samples = gini_networks[0][i][0]  # 1D array
        poorest_samples = poorest_networks[0][i][0]  # 1D array
        richest_samples = richest_networks[0][i][0]  # 1D array
        mean_gini, lower_gini, upper_gini = calc_bounds_1d(gini_samples, 0.95)
        mean_poorest, _, _ = calc_bounds_1d(poorest_samples, 0.95)
        mean_richest, _, _ = calc_bounds_1d(richest_samples, 0.95)
        row_titles.append(
            f"a Beta distribution, Expenditure = {np.round(a, 5)}, Gini = {np.round(mean_gini, 5)}, Poorest Prop= {np.round(mean_poorest, 5)}, Richest Prop= {np.round(mean_richest, 5)} "
        )


    plot_means_end_points_emissions_confidence_split_gradient(fileName, emissions_networks, property_values_list_col, property_values_list_row,network_titles,row_titles, name)
    plot_means_end_points_emissions_confidence_split_gradient_alt(fileName, emissions_networks, property_values_list_col, property_values_list_row,network_titles,row_titles, name)
    plot_emissions_vs_gini_scatter(fileName, emissions_networks, gini_networks, property_values_list_col, property_values_list_row,network_titles, name)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/network_ineq_tau_16_47_17__18_06_2025"#network_ineq_tau_00_14_13__18_06_2025"#network_ineq_tau_17_24_33__17_06_2025"#network_ineq_tau_11_50_31__17_06_2025"#network_ineq_tau_10_27_38__17_06_2025"#network_ineq_tau_11_59_40__11_06_2025"
    )