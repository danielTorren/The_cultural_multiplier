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
N = 500  # Number of individuals
M = 3  # Number of services
a_values = np.linspace(0.1, 8, 50)  # Beta distribution parameter a
b_values = np.linspace(0.1, 8, 50)  # Beta distribution parameter b
sector_preferences = 1/M 
prices_low_carbon_m = 1  # Prices for low-carbon goods set to 1
prices_high_carbon_instant = 1  # Prices for high-carbon goods set to 1
low_carbon_substitutability = 2  # sigma = 2
sector_substitutability = 2  # nu = 2
instant_expenditure = 1/N  # Expenditure set to 1 for all individuals

M_values = [3, 30, 300]  # Different M values

# Initialize the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loop over different M values
for k, M in enumerate(M_values):
    # Adjust sector preferences and prices to match M
    sector_preferences = 1/M
    
    # Initialize results storage
    emissions_results = np.zeros((len(a_values), len(b_values)))

    # Compute emissions (sum(H)) for each combination of a and b
    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            # Generate the preference matrix A using the beta distribution
            low_carbon_preference_matrix = np.random.beta(a, b, (N, M))
            
            # Instantiate the calculator
            calculator = NCESUtilityCalculator(low_carbon_preference_matrix, sector_preferences, prices_low_carbon_m, prices_high_carbon_instant, low_carbon_substitutability, sector_substitutability, instant_expenditure)
            
            # Calculate H and L
            H_m_matrix, L_m_matrix = calculator._calc_consumption()
            
            # Calculate total emissions (sum(H))
            emissions_results[i, j] = np.sum(H_m_matrix)

    # Plotting the contour for the current M
    contour = axes[k].contourf(b_values, a_values, emissions_results, cmap='viridis')
    axes[k].set_title(f'M = {M}')
    axes[k].set_xlabel('Beta distribution parameter b')
    if k == 0:
        axes[k].set_ylabel('Beta distribution parameter a')
    fig.colorbar(contour, ax=axes[k])

plt.tight_layout()
plt.show()