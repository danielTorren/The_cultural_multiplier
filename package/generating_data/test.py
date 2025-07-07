import numpy as np
import matplotlib.pyplot as plt

def plot_all_weightings(p1, grid_size=100, theta=5, M=2):
    """
    Plot all four weighting schemes in one row of subplots:
    - Softmax Cultural Multiplier
    - Cosine Similarity (shifted/rescaled)
    - Euclidean distance
    - Cosine Similarity (non-shifted)
    """

    # -----------------------------------------------------
    # common grid
    p2_x = np.linspace(0, 1, grid_size)
    p2_y = np.linspace(0, 1, grid_size)
    P2_X, P2_Y = np.meshgrid(p2_x, p2_y)
    p2s = np.vstack([P2_X.ravel(), P2_Y.ravel()]).T

    # -----------------------------------------------------
    # Softmax Cultural Multiplier (identity difference)
    I_i = np.mean(p1)
    I_j = np.mean(p2s, axis=1)
    diffs = np.abs(I_i - I_j)
    weights_cm = np.exp(-theta * diffs)
    weights_cm /= np.sum(weights_cm)
    weights_cm_grid = weights_cm.reshape(grid_size, grid_size)

    # -----------------------------------------------------
    # Cosine Similarity (shifted/rescaled)
    shifted_p1 = 2 * p1 - 1
    shifted_p2s = 2 * p2s - 1

    norm_shifted_p1 = np.linalg.norm(shifted_p1)
    if norm_shifted_p1 == 0:
        norm_shifted_p1 = 1e-6
    norm_shifted_p2s = np.linalg.norm(shifted_p2s, axis=1)
    norm_shifted_p2s[norm_shifted_p2s == 0] = 1e-6

    dot_products = shifted_p2s @ shifted_p1
    cos_sims_shifted = dot_products / (norm_shifted_p1 * norm_shifted_p2s)
    rescaled_cos_sims = (cos_sims_shifted + 1) / 2

    weights_cos_shifted = np.exp(theta * rescaled_cos_sims)
    weights_cos_shifted /= np.sum(weights_cos_shifted)
    weights_cos_shifted_grid = weights_cos_shifted.reshape(grid_size, grid_size)

    # -----------------------------------------------------
    # Euclidean distance
    diffs = p2s - p1
    dists = np.linalg.norm(diffs, axis=1)
    max_possible_dist = np.sqrt(M)
    dists_normalized = dists / max_possible_dist

    weights_euc = np.exp(-theta * dists_normalized)
    weights_euc /= np.sum(weights_euc)
    weights_euc_grid = weights_euc.reshape(grid_size, grid_size)

    # -----------------------------------------------------
    # Cosine Similarity (non-shifted)
    norm_p1 = np.linalg.norm(p1)
    if norm_p1 == 0:
        norm_p1 = 1e-6
    norm_p2s = np.linalg.norm(p2s, axis=1)
    norm_p2s[norm_p2s == 0] = 1e-6

    dot_products_ns = p2s @ p1
    cos_sims_ns = dot_products_ns / (norm_p1 * norm_p2s)
    rescaled_cos_sims_ns = (cos_sims_ns + 1) / 2  # optional rescale to [0,1]

    weights_cos_ns = np.exp(theta * rescaled_cos_sims_ns)
    weights_cos_ns /= np.sum(weights_cos_ns)
    weights_cos_ns_grid = weights_cos_ns.reshape(grid_size, grid_size)

    # -----------------------------------------------------
    # Plot all four in a single row of subplots
    vmin = min(weights_cm_grid.min(), weights_cos_shifted_grid.min(),
               weights_euc_grid.min(), weights_cos_ns_grid.min())
    vmax = max(weights_cm_grid.max(), weights_cos_shifted_grid.max(),
               weights_euc_grid.max(), weights_cos_ns_grid.max())

    fig, axes = plt.subplots(1, 4, figsize=(24,6), constrained_layout=True)

    im0 = axes[0].imshow(weights_cm_grid, extent=[0,1,0,1], origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Softmax")
    axes[0].set_xlabel("Preference Sector 1")
    axes[0].set_ylabel("Preference Sector 2")
    axes[0].text(
        0.5, -0.25,
        r"$\alpha^{SM}_{i,j} = \frac{ \exp\left( -\theta |I_i - I_j| \right) }{ \sum }$",
        ha="center", va="center", transform=axes[0].transAxes, fontsize=11
    )

    im1 = axes[1].imshow(weights_cos_shifted_grid, extent=[0,1,0,1], origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Cosine Similarity (shifted)")
    axes[1].set_xlabel("Preference Sector 1")
    axes[1].set_ylabel("Preference Sector 2")
    axes[1].text(
        0.5, -0.25,
        r"$\alpha^{cos}_{i,j} = \frac{ \exp\left( \theta \, \frac{\cos(2p_i-1,2p_j-1)+1}{2} \right) }{ \sum }$",
        ha="center", va="center", transform=axes[1].transAxes, fontsize=11
    )

    im2 = axes[2].imshow(weights_euc_grid, extent=[0,1,0,1], origin="lower", vmin=vmin, vmax=vmax)
    axes[2].set_title("Euclidean")
    axes[2].set_xlabel("Preference Sector 1")
    axes[2].set_ylabel("Preference Sector 2")
    axes[2].text(
        0.5, -0.25,
        r"$\alpha^{E}_{i,j} = \frac{ \exp\left( -\theta \, \frac{||p_i - p_j||}{\sqrt{M}} \right) }{ \sum }$",
        ha="center", va="center", transform=axes[2].transAxes, fontsize=11
    )

    im3 = axes[3].imshow(weights_cos_ns_grid, extent=[0,1,0,1], origin="lower", vmin=vmin, vmax=vmax)
    axes[3].set_title("Cosine Similarity (non-shifted)")
    axes[3].set_xlabel("Preference Sector 1")
    axes[3].set_ylabel("Preference Sector 2")
    axes[3].text(
        0.5, -0.25,
        r"$\alpha^{cos}_{i,j} = \frac{ \exp\left( \theta \, \frac{\cos(p_i,p_j)+1}{2} \right) }{ \sum }$",
        ha="center", va="center", transform=axes[3].transAxes, fontsize=11
    )

    # shared colorbar
    cbar = fig.colorbar(im3, ax=axes.ravel(), shrink=0.8)
    cbar.set_label("Interaction Strength")

    plt.show()

# ------------------ Example call ------------------

if __name__ == "__main__":
    p1 = np.array([0.5, 0.5])
    theta = 5
    grid_size = 100

    plot_all_weightings(p1, grid_size, theta, M=2)
