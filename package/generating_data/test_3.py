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

M_values = [3, 30, 300]  # Different M values

# Initialize the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loop over different M values
for k, M in enumerate(M_values):
    # Adjust the sector_preferences and prices_low_carbon_m to match M
    sector_preferences = np.ones(M)
    prices_low_carbon_m = np.ones(M)
    
    # Initialize results storage
    ratio_results = np.zeros((len(a_values), len(b_values)))

    # Compute the ratio sum(L) / (sum(L) + sum(H)) for each combination of a and b
    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            # Generate the preference matrix A using the beta distribution
            low_carbon_preference_matrix = np.random.beta(a, b, (N, M))
            calculator = NCESUtilityCalculator(low_carbon_preference_matrix, sector_preferences, prices_low_carbon_m, prices_high_carbon_instant, low_carbon_substitutability, sector_substitutability, instant_expenditure)
            H_m_matrix, L_m_matrix = calculator._calc_consumption()

            # Calculate the ratio and store it
            ratio_results[i, j] = np.sum(L_m_matrix) / (np.sum(L_m_matrix) + np.sum(H_m_matrix))

    # Plotting the contour for the current M
    contour = axes[k].contourf(b_values, a_values, ratio_results, cmap='coolwarm')
    axes[k].set_title(f'M = {M}')
    axes[k].set_xlabel('Beta distribution parameter b')
    if k == 0:
        axes[k].set_ylabel('Beta distribution parameter a')
    fig.colorbar(contour, ax=axes[k])

plt.tight_layout()
plt.show()