import numpy as np
import matplotlib.pyplot as plt

# Initialize the NCESUtilityCalculator class

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
        omega_vector = ((self.prices_high_carbon_instant* self.low_carbon_preference_matrix) / (self.prices_low_carbon_m * (1 - self.low_carbon_preference_matrix))) ** self.low_carbon_substitutability
        return omega_vector

    def _calc_chi_m_nested_CES(self, Omega_m_matrix):
        chi_m = (((self.sector_preferences * self.low_carbon_preference_matrix)/(self.prices_low_carbon_m*Omega_m_matrix**(1/self.low_carbon_substitutability)))*(self.low_carbon_preference_matrix*Omega_m_matrix**((self.low_carbon_substitutability-1)/(self.low_carbon_substitutability)) + 1 - self.low_carbon_preference_matrix)**((self.sector_substitutability-self.low_carbon_substitutability)/(self.sector_substitutability*(self.low_carbon_substitutability-1))))** self.sector_substitutability
        return chi_m

    def _calc_Z(self,Omega_m_matrix, chi_m_tensor):
        common_vector = Omega_m_matrix * self.prices_low_carbon_m + self.prices_high_carbon_instant
        no_sum_Z_terms = chi_m_tensor * common_vector
        Z = no_sum_Z_terms.sum(axis=1)#sum with people
        return Z

    def _calc_consumption_quantities_nested_CES(self):
        Omega_m_matrix = self._calc_Omega_m()
        chi_m_tensor = self._calc_chi_m_nested_CES(Omega_m_matrix)
        Z_vec = self._calc_Z(Omega_m_matrix, chi_m_tensor)

        Z_matrix = np.tile(Z_vec, (self.low_carbon_preference_matrix.shape[1], 1)).T#repeat it so that can have chi tensor
        H_m_matrix = self.instant_expenditure * chi_m_tensor / Z_matrix
        L_m_matrix = Omega_m_matrix * H_m_matrix

        return H_m_matrix, L_m_matrix

    def _calc_consumption(self):
        H_m_matrix, L_m_matrix = self._calc_consumption_quantities_nested_CES()
        return H_m_matrix, L_m_matrix
    
# Parameters
N = 100  # Number of individuals
M_values = [3, 30, 300]  # Different numbers of services
a, b = 1, 1  # Parameters for the beta distribution
instant_expenditure = 1 / N  # Expenditure set to 1/N for all agents
prices_low_carbon_m = 1
prices_high_carbon_instant = 1
low_carbon_substitutability = 2  # sigma = 2
sector_substitutability = 2  # nu = 2

# Initialize the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loop over different M values
for k, M in enumerate(M_values):
    sector_preferences = 1 / M  # Sector preferences for M services
    
    # Generate A values using a beta distribution
    A_values = np.random.beta(a, b, size=(N, M))
    
    # Calculate L and H for these A values
    calculator = NCESUtilityCalculator(A_values, sector_preferences, prices_low_carbon_m, prices_high_carbon_instant, low_carbon_substitutability, sector_substitutability, instant_expenditure)
    H_m_matrix, L_m_matrix = calculator._calc_consumption()

    # Scatter plot of A vs. H and A vs. L
    axes[k].scatter(A_values.flatten(), H_m_matrix.flatten(), color='red', label='High-carbon goods (H_m)', alpha=0.6)
    axes[k].scatter(A_values.flatten(), L_m_matrix.flatten(), color='green', label='Low-carbon goods (L_m)', alpha=0.6)

    axes[k].set_xlabel('Preference for Low-carbon Goods (A)')
    axes[k].set_title(f'Sectors = {M}')
    axes[k].legend()
    axes[k].grid(True)

axes[0].set_ylabel('Quantity')
plt.tight_layout()
plt.show()