"""Define individual agent class
A module that defines "individuals" that have vectors of attitudes towards behaviours whose evolution
is determined through weighted social interactions.



Created: 10/10/2022
"""

# imports
import numpy as np
import numpy.typing as npt

# modules
class Individual:

    """
    Class to represent individuals with identities, preferences and consumption

    """

    def __init__(
        self,
        individual_params,
        low_carbon_preferences,
        service_preferences,
        budget,
        id_n,
    ):

        self.low_carbon_preferences = low_carbon_preferences

        self.service_preferences = service_preferences
        
        self.init_budget = budget
        self.instant_budget = self.init_budget

        self.carbon_price = individual_params["carbon_price"]

        self.M = individual_params["M"]
        self.t = individual_params["t"]
        self.save_timeseries_data = individual_params["save_timeseries_data"]
        self.compression_factor = individual_params["compression_factor"]
        self.phi_array = individual_params["phi_array"]
        self.service_substitutability = individual_params["service_substitutability"]
        self.low_carbon_substitutability_array = individual_params["low_carbon_substitutability"]
        self.prices_low_carbon = individual_params["prices_low_carbon"]
        self.prices_high_carbon = individual_params["prices_high_carbon"]
        self.clipping_epsilon = individual_params["clipping_epsilon"]
        self.ratio_preference_or_consumption_identity = individual_params["ratio_preference_or_consumption_identity"]

        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price

        self.id = id_n

        self.Omega_m = self.calc_omega()
        self.chi_m = self.calc_chi_m()
        self.H_m, self.L_m = self.calc_consumption_quantities()

        self.identity = self.calc_identity()
        self.initial_carbon_emissions = self.calc_total_emissions()
        self.flow_carbon_emissions = self.initial_carbon_emissions

        self.utility = self.calc_utility()

        if self.save_timeseries_data:
            self.history_low_carbon_preferences = [self.low_carbon_preferences]
            self.history_identity = [self.identity]
            self.history_flow_carbon_emissions = [self.flow_carbon_emissions]
            self.history_utility = [self.utility]

    def calc_omega(self):        
        omega_vector = ((self.prices_high_carbon_instant*self.low_carbon_preferences)/(self.prices_low_carbon*(1- self.low_carbon_preferences )))**(self.low_carbon_substitutability_array)
        return omega_vector
    
    def calc_chi_m(self):
        power_omega = (self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array
        power_second = (self.service_substitutability-1)*((self.low_carbon_substitutability_array)/((self.low_carbon_substitutability_array - 1)))
        
        first_bit = (self.service_preferences/self.prices_high_carbon_instant)**(self.service_substitutability)
        second_bit = (self.low_carbon_preferences*(self.Omega_m**power_omega) + (1 - self.low_carbon_preferences))**power_second
        
        chi_components = first_bit*second_bit

        return chi_components
         
    def calc_consumption_quantities(self):
        common_vector_denominator = self.Omega_m*self.prices_low_carbon + self.prices_high_carbon_instant

        H_m_denominators = np.matmul(self.chi_m, common_vector_denominator)

        H_m = self.instant_budget*self.chi_m/H_m_denominators
        L_m = H_m*self.Omega_m

        ###NOT SURE I NEED THE LINE BELOW
        H_m_clipped = np.clip(H_m, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)
        L_m_clipped = np.clip(L_m, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)
        
        return H_m_clipped,L_m_clipped
        #return H_m,L_m

    def calc_identity(self) -> float:
        if self.ratio_preference_or_consumption_identity == 1.0:
            identity = np.mean(self.low_carbon_preferences)
        elif self.ratio_preference_or_consumption_identity == 0.0:
            identity = np.mean(self.consumption_ratio)
        elif self.ratio_preference_or_consumption_identity > 0.0 and self.ratio_preference_or_consumption_identity < 1.0:
            identity = self.ratio_preference_or_consumption_identity*np.mean(self.low_carbon_preferences) + (1-self.ratio_preference_or_consumption_identity)*np.mean(self.consumption_ratio)
        else:
            raise("Invalid ratio_preference_or_consumption_identity = [0,1]", self.ratio_preference_or_consumption_identity)
        return identity

    def update_preferences(self, social_component):
        low_carbon_preferences = (1 - self.phi_array)*self.low_carbon_preferences + (self.phi_array)*(social_component)
        
        ###NOT SURE I NEED THE LINE BELOW
        self.low_carbon_preferences  = np.clip(low_carbon_preferences, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)#this stops the guassian error from causing A to be too large or small thereby producing nans
        #self.low_carbon_preferences = low_carbon_preferences

    def calc_total_emissions(self):      
        return sum(self.H_m)
    
    def calc_utility(self):
        psuedo_utility = (self.low_carbon_preferences*(self.L_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)) + (1 - self.low_carbon_preferences)*(self.H_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)))**(self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))
        sum_U = (sum(self.service_preferences*(psuedo_utility**((self.service_substitutability -1)/self.service_preferences))))**(self.service_preferences/(self.service_preferences-1))
        return sum_U
    
    def calc_consumption_ratio(self):
        return self.L_m/(self.L_m + self.H_m)

    def save_timeseries_data_individual(self):
        """
        Save time series data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.history_low_carbon_preferences.append(self.low_carbon_preferences)
        self.history_identity.append(self.identity)
        self.history_flow_carbon_emissions.append(self.flow_carbon_emissions)
        self.history_utility.append(self.utility)

    def next_step(self, t: int, social_component: npt.NDArray, carbon_dividend, carbon_price):

        self.t = t

        #update prices
        self.carbon_price = carbon_price
        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price
        #update_budget
        self.instant_budget = self.init_budget + carbon_dividend

        #print("self.instant_budget", self.instant_budget, carbon_dividend ,self.init_budget)
        
        #update preferences 
        self.update_preferences(social_component)
        

        #calculate consumption
        self.Omega_m = self.calc_omega()
        self.chi_m = self.calc_chi_m()
        self.H_m, self.L_m = self.calc_consumption_quantities()
        self.consumption_ratio = self.calc_consumption_ratio()

        #calc_identity
        self.identity = self.calc_identity()

        #calc_emissions
        self.flow_carbon_emissions = self.calc_total_emissions()
        
        #calc_utility
        self.utility = self.calc_utility()

        if (self.save_timeseries_data) and (self.t % self.compression_factor == 0):
            #calc utility
            
            self.save_timeseries_data_individual()
