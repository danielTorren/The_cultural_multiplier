import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        H_m_matrix = self.instant_expenditure[:, np.newaxis] * chi_m_tensor / Z_matrix
        L_m_matrix = Omega_m_matrix * H_m_matrix

        return H_m_matrix, L_m_matrix

    def _calc_consumption(self):
        H_m_matrix, L_m_matrix = self._calc_consumption_quantities_nested_CES()
        return H_m_matrix, L_m_matrix


# Parameters
N = 500  # Number of individuals
M = 3  # Number of services
a_values = np.linspace(0.1, 8, 50)  # Beta distribution parameter a
b_values = np.linspace(0.1, 8, 50)  # Beta distribution parameter b
sector_preferences = np.ones(M)  # All sector preferences set to 1
prices_low_carbon_m = np.ones(M)  # Prices for low-carbon goods set to 1
prices_high_carbon_instant = 1  # Prices for high-carbon goods set to 1
low_carbon_substitutability = 2  # sigma = 2
sector_substitutability = 2  # nu = 2
instant_expenditure = np.ones(N)  # Expenditure set to 1 for all individuals

# Initialize results storage
H_m_results = np.zeros((len(a_values), len(b_values)))
L_m_results = np.zeros((len(a_values), len(b_values)))

# Compute H and L for each combination of a and b
for i, a in enumerate(a_values):
    for j, b in enumerate(b_values):
        # Generate the preference matrix A using the beta distribution
        low_carbon_preference_matrix = np.random.beta(a, b, (N, M))
        calculator = NCESUtilityCalculator(low_carbon_preference_matrix, sector_preferences, prices_low_carbon_m, prices_high_carbon_instant, low_carbon_substitutability, sector_substitutability, instant_expenditure)
        H_m_matrix, L_m_matrix = calculator._calc_consumption()

        # Store the mean value of the quantities across all individuals
        H_m_results[i, j] = np.sum(H_m_matrix)
        L_m_results[i, j] = np.sum(L_m_matrix)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Contour plot for H_m
contour_1 = axes[0].contourf(b_values, a_values, H_m_results, cmap='Reds')
axes[0].set_title('High-carbon goods (H_m)')
axes[0].set_xlabel('Beta distribution parameter b')
axes[0].set_ylabel('Beta distribution parameter a')
fig.colorbar(contour_1, ax=axes[0])

# Contour plot for L_m
contour_2 = axes[1].contourf(b_values, a_values, L_m_results, cmap='Greens')
axes[1].set_title('Low-carbon goods (L_m)')
axes[1].set_xlabel('Beta distribution parameter b')
axes[1].set_ylabel('Beta distribution parameter a')
fig.colorbar(contour_2, ax=axes[1])

plt.tight_layout()
plt.show()
