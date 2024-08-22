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
N = 100
M_values = [3, 30, 300]  # Different numbers of services
low_carbon_preference_values = np.linspace(0, 1, 100)  # A values ranging from 0 to 1
sector_substitutability = 2  # nu = 2
instant_expenditure = 1/N  # Expenditure set to 1 for all agents

prices_low_carbon_m = 1
prices_high_carbon_instant = 1
low_carbon_substitutability = 2  # sigma = 2
# Initialize the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loop over different M values
for k, M in enumerate(M_values):
    # Adjust sector preferences and prices to match M
    sector_preferences = 1/M
    # Store results for plotting
    H_m_results = []
    L_m_results = []

    # Calculate and store L and H for each A
    for A in low_carbon_preference_values:
        low_carbon_preference_matrix = np.full((N, M), A)
        #print("low_carbon_preference_matrix",low_carbon_preference_matrix)
        #quit()
        calculator = NCESUtilityCalculator(low_carbon_preference_matrix, sector_preferences, prices_low_carbon_m, prices_high_carbon_instant, low_carbon_substitutability, sector_substitutability, instant_expenditure)
        H_m_matrix, L_m_matrix = calculator._calc_consumption()
        
        # Aggregate the results for all services and agents
        H_m_results.append(np.sum(H_m_matrix))
        L_m_results.append(np.sum(L_m_matrix))
    
    # Plotting for the current M
    axes[k].plot(low_carbon_preference_values, H_m_results, label='High-carbon goods (H_m)', color='red')
    axes[k].plot(low_carbon_preference_values, L_m_results, label='Low-carbon goods (L_m)', color='green')
    axes[k].set_xlabel('Preference for Low-carbon Goods (A)')
    
    axes[k].set_title(f'Sectors = {M}')
    axes[k].legend()
    axes[k].grid(True)
axes[0].set_ylabel('Quantity')
plt.tight_layout()
plt.show()