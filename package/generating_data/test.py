import numpy as np
import matplotlib.pyplot as plt

# Parameters
grid_size = 100
theta = 5
M = 2  # number of sectors

# Grid of agent j's preferences
p2_x = np.linspace(0, 1, grid_size)
p2_y = np.linspace(0, 1, grid_size)
P2_X, P2_Y = np.meshgrid(p2_x, p2_y)
p2s = np.vstack([P2_X.ravel(), P2_Y.ravel()]).T

# Agent i's fixed preference
p1 = np.array([0.5, 0.5])

# Compute identity of agent i
I_i = np.mean(p1)

# Compute identity of agent j's preferences on the grid
I_j = np.mean(p2s, axis=1)

# Absolute differences in identity
diffs = np.abs(I_i - I_j)

# Cultural multiplier weights (unnormalized)
unnormalized_weights = np.exp(-theta * diffs)
weights = unnormalized_weights / np.sum(unnormalized_weights)
weights_grid = weights.reshape(grid_size, grid_size)

# Cosine similarity between p1 and each p2
norm_p1 = np.linalg.norm(p1)
norms_p2 = np.linalg.norm(p2s, axis=1)
norms_p2[norms_p2 == 0] = 1e-6  # avoid division by zero
dot_products = p2s @ p1
cos_sims = dot_products / (norm_p1 * norms_p2)

unnormalized_cosine_weights = np.exp(theta * cos_sims)
cosine_weights = unnormalized_cosine_weights / np.sum(unnormalized_cosine_weights)
cosine_weights_grid = cosine_weights.reshape(grid_size, grid_size)

# determine shared vmin/vmax
vmin = min(weights_grid.min(), cosine_weights_grid.min())
vmax = max(weights_grid.max(), cosine_weights_grid.max())

# ---------- FIRST FIGURE: heatmaps of the weighting matrices ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

im0 = axes[0].imshow(weights_grid, extent=[0,1,0,1], origin="lower", vmin=vmin, vmax=vmax)
axes[0].set_title(r"Softmax Cultural Multiplier")
axes[0].set_xlabel("Preference Sector 1")
axes[0].set_ylabel("Preference Sector 2")

im1 = axes[1].imshow(cosine_weights_grid, extent=[0,1,0,1], origin="lower", vmin=vmin, vmax=vmax)
axes[1].set_title(r"Softmax Cultural Cosine")
axes[1].set_xlabel("Preference Sector 1")
axes[1].set_ylabel("Preference Sector 2")

# colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label("Interaction Strength")

# Add equations below
axes[0].text(
    0.5, -0.15,
    r"$\alpha^{CM}_{i,j} = \frac{ \exp\left( -\theta | I_i - I_j | \right) }{ \sum_j \exp\left( -\theta | I_i - I_j | \right) }$",
    ha="center", va="center", transform=axes[0].transAxes, fontsize=11
)

axes[1].text(
    0.5, -0.15,
    r"$\alpha^{cos}_{i,j} = \frac{ \exp\left( \theta \, \mathrm{cos\_sim}(p_i, p_j) \right) }{ \sum_j \exp\left( \theta \, \mathrm{cos\_sim}(p_i, p_j) \right) }$",
    ha="center", va="center", transform=axes[1].transAxes, fontsize=11
)

plt.tight_layout(rect=[0, 0, 0.9, 1])


# ---------- SECOND FIGURE: raw terms ----------
diffs_grid = diffs.reshape(grid_size, grid_size)
cos_sims_grid = cos_sims.reshape(grid_size, grid_size)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))

im2 = axes2[0].imshow(-diffs_grid, extent=[0,1,0,1], origin="lower")
axes2[0].set_title(r"Absolute Identity Difference $|I_i - I_j|$")
axes2[0].set_xlabel("Preference Sector 1")
axes2[0].set_ylabel("Preference Sector 2")

im3 = axes2[1].imshow(cos_sims_grid, extent=[0,1,0,1], origin="lower")
axes2[1].set_title(r"Cosine Similarity $\mathrm{cos\_sim}(p_i, p_j)$")
axes2[1].set_xlabel("Preference Sector 1")
axes2[1].set_ylabel("Preference Sector 2")

# shared colorbar for second figure
cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
cbar2 = fig2.colorbar(im3, cax=cbar_ax2)
cbar2.set_label("Value")

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()
