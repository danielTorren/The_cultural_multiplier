import numpy as np
import matplotlib.pyplot as plt

class NCESUtilityCalculator:
    def __init__(self, low_carbon_preference_matrix, sector_preferences, prices_low_carbon_m, prices_high_carbon_instant, low_carbon_substitutability, sector_substitutability, instant_expenditure):
        self.low_carbon_preference_matrix = low_carbon_preference_matrix
        self.sector_preferences = sector_preferences
        self.prices_low_carbon_m = prices_low_carbon_m
        self.prices_high_carbon_instant = prices_high_carbon_instant
        self.low_carbon_substitutability = low_carbon_substitutability
        self.sector_substitutability = sector_substitutability
        self.instant_expenditure = instant_expenditure

    def _calc_Omega_m(self):
        term_1 = self.prices_high_carbon_instant * self.low_carbon_preference_matrix
        term_2 = self.prices_low_carbon_m * (1 - self.low_carbon_preference_matrix)
        omega_vector = (term_1 / term_2) ** self.low_carbon_substitutability
        return omega_vector

    def _calc_n_tilde_m(self):
        Omega_m_matrix = self._calc_Omega_m()
        n_tilde_m = (self.low_carbon_preference_matrix * (Omega_m_matrix ** ((self.low_carbon_substitutability - 1) / self.low_carbon_substitutability)) +
                     (1 - self.low_carbon_preference_matrix)) ** (self.low_carbon_substitutability / (self.low_carbon_substitutability - 1))
        return n_tilde_m

    def _calc_chi_m_nested_CES(self):
        n_tilde_m_matrix = self._calc_n_tilde_m()
        chi_m = ((self.sector_preferences * (n_tilde_m_matrix ** ((self.sector_substitutability - 1) / self.sector_substitutability))) /
                 self.prices_high_carbon_instant) ** self.sector_substitutability
        return chi_m

    def _calc_Z(self):
        Omega_m_matrix = self._calc_Omega_m()
        chi_m_tensor = self._calc_chi_m_nested_CES()
        common_vector = Omega_m_matrix * self.prices_low_carbon_m + self.prices_high_carbon_instant
        no_sum_Z_terms = chi_m_tensor * common_vector
        Z = no_sum_Z_terms.sum(axis=1)
        return Z

    def _calc_consumption_quantities_nested_CES(self):
        Z_vec = self._calc_Z()
        chi_m_tensor = self._calc_chi_m_nested_CES()
        Omega_m_matrix = self._calc_Omega_m()

        Z_matrix = np.tile(Z_vec, (self.low_carbon_preference_matrix.shape[1], 1)).T
        H_m_matrix = self.instant_expenditure * chi_m_tensor / Z_matrix
        L_m_matrix = Omega_m_matrix * H_m_matrix

        return H_m_matrix, L_m_matrix

    def _calc_consumption(self):
        H_m_matrix, L_m_matrix = self._calc_consumption_quantities_nested_CES()
        return H_m_matrix, L_m_matrix
    
# Parameters
M = 2  # Number of sectors
N = 100  # Number of individuals
sector_preferences =1 / M  # Sector preferences set to 1/M for each sector
prices_low_carbon_m = 1  # Prices for low-carbon goods set to 1 for each sector
prices_high_carbon_instant = 1  # Prices for high-carbon goods set to 1 for each sector
low_carbon_substitutability =  2  # sigma = 2 for each sector
sector_substitutability = 2  # nu = 2
instant_expenditure = 1 / N  # Expenditure set to 1/N for each agent

# Create a grid of A_1 and A_2 values
A1_values = np.linspace(0, 1, 100)
A2_values = np.linspace(0, 1, 100)
A1_grid, A2_grid = np.meshgrid(A1_values, A2_values)

# Initialize matrices to store results
H_m_results = np.zeros_like(A1_grid)
L_m_results = np.zeros_like(A1_grid)

# Calculate L and H for each pair of (A1, A2)
for i in range(A1_grid.shape[0]):
    for j in range(A1_grid.shape[1]):
        A1 = A1_grid[i, j]
        A2 = A2_grid[i, j]
        
        low_carbon_preference_matrix = np.array([[A1, A2]] * N)
        calculator = NCESUtilityCalculator(low_carbon_preference_matrix, sector_preferences, prices_low_carbon_m, prices_high_carbon_instant, low_carbon_substitutability, sector_substitutability, instant_expenditure)
        H_m_matrix, L_m_matrix = calculator._calc_consumption()
        
        # Sum the results for all agents
        H_m_results[i, j] = np.sum(H_m_matrix)
        L_m_results[i, j] = np.sum(L_m_matrix)

# Plot the contour maps
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Contour plot for H quantities
contour_H = axes[0].contourf(A1_grid, A2_grid, H_m_results, cmap='Reds', levels=20)
axes[0].set_xlabel('Preference for Low-carbon Goods in Sector 1 (A1)')
axes[0].set_ylabel('Preference for Low-carbon Goods in Sector 2 (A2)')
axes[0].set_title('High-carbon Goods Quantity (H) Contour Plot')
fig.colorbar(contour_H, ax=axes[0])

# Contour plot for L quantities
contour_L = axes[1].contourf(A1_grid, A2_grid, L_m_results, cmap='Greens', levels=20)
axes[1].set_xlabel('Preference for Low-carbon Goods in Sector 1 (A1)')
axes[1].set_ylabel('Preference for Low-carbon Goods in Sector 2 (A2)')
axes[1].set_title('Low-carbon Goods Quantity (L) Contour Plot')
fig.colorbar(contour_L, ax=axes[1])

plt.tight_layout()
plt.show()
