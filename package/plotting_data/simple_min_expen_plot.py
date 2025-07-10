import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from package.resources.utility import load_object

def calc_bounds(data, confidence=0.95):
    """Calculate mean and confidence interval for 1D array."""
    data = np.array(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    z = norm.ppf(0.5 + confidence / 2)
    lower = mean - z * sem
    upper = mean + z * sem
    return mean, lower, upper

def plot_emissions_vs_min_expenditure(fileName):
    # Load data
    emissions_data = load_object(fileName + "/Data", "emissions_data_min_expenditure")
    emissions_data_ref = load_object(fileName + "/Data", "emissions_data_min_expenditure_ref")
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    min_expenditure_shares = variable_parameters_dict["property_vals"]
    network_labels = ["Small-World", "SBM", "Scale-Free"]

    num_networks = len(emissions_data)
    reps = emissions_data.shape[1]
    seeds = emissions_data.shape[2]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(num_networks):
        means = []
        lowers = []
        uppers = []
        for j in range(len(min_expenditure_shares)):
            emissions = emissions_data[i][j].flatten()  # Combine across seeds
            mu, lo, hi = calc_bounds(emissions)
            means.append(mu)
            lowers.append(lo)
            uppers.append(hi)

        ax.plot(min_expenditure_shares, means, label=network_labels[i], marker='o')
        ax.fill_between(min_expenditure_shares, lowers, uppers, alpha=0.2)

    ax.set_xlabel("Minimum expenditure share (h_min ⋅ prices)", fontsize=12)
    ax.set_ylabel("Cumulative carbon emissions, E", fontsize=12)
    ax.set_title("Emissions vs. Minimum Expenditure Share", fontsize=14)
    ax.legend(title="Network Structure")
    ax.grid(True)
    fig.tight_layout()

    fig.savefig(f"{fileName}/Plots/emissions_vs_min_expenditure.png", dpi=300)
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_emissions_vs_min_expenditure("results/simple_h_min_expenditure_12_40_00__10_07_2025")  # adjust path as needed
