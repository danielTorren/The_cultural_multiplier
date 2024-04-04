"""Define individual agent class
A module that defines "individuals" that have vectors of attitudes towards behaviours whose evolution
is determined through weighted social interactions.

Created: 10/10/2022
"""

# imports
from logging import raiseExceptions
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
        expenditure,
        low_carbon_substitutability_array,
        id
    ):

        self.low_carbon_preferences_init = low_carbon_preferences   
        self.low_carbon_preferences = low_carbon_preferences  
        self.init_expenditure = expenditure
        self.instant_expenditure = self.init_expenditure

        self.M = individual_params["M"]
        self.t = individual_params["t"]
        self.save_timeseries_data_state = individual_params["save_timeseries_data_state"]
        self.compression_factor_state = individual_params["compression_factor_state"]
        self.imitation_state = individual_params["imitation_state"]
        self.phi_array = individual_params["phi_array"]
        self.sector_preferences = individual_params["sector_preferences"]
        self.low_carbon_substitutability_array = low_carbon_substitutability_array
        #self.low_carbon_substitutability_array = individual_params["low_carbon_substitutability"]
        self.prices_low_carbon_m = individual_params["prices_low_carbon_m"]
        self.prices_high_carbon_m = individual_params["prices_high_carbon_m"]
        self.clipping_epsilon = individual_params["clipping_epsilon"]
        self.burn_in_duration = individual_params["burn_in_duration"]
        self.alpha_change_state = individual_params["alpha_change_state"]
        self.static_internal_preference_state = individual_params["static_internal_preference_state"]

        self.sector_substitutability = individual_params["sector_substitutability"]

        self.update_prices(individual_params["init_carbon_price_m"])
        #self.prices_high_carbon_instant = self.prices_high_carbon_m + individual_params["init_carbon_price_m"]

        self.id = id

        #update_consumption
        self.update_consumption()
        
        self.identity = self.calc_identity()

        self.initial_carbon_emissions = self.calc_total_emissions()
        self.flow_carbon_emissions = self.initial_carbon_emissions
        self.utility,self.pseudo_utility = self.calc_utility_nested_CES()

        #print("self.t",self.t, self.burn_in_duration)
        #if self.t == self.burn_in_duration and self.save_timeseries_data_state:
        #    self.set_up_time_series()
    
    def set_up_time_series(self):
        self.history_low_carbon_preferences = [self.low_carbon_preferences]
        self.history_omega_m = [self.Omega_m]
        self.history_chi_m = [self.chi_m]
        self.history_identity = [self.identity]
        self.history_flow_carbon_emissions = [self.flow_carbon_emissions]
        self.history_utility = [self.utility]
        self.history_pseudo_utility = [self.pseudo_utility]
        self.history_H_m = [self.H_m]
        self.history_L_m = [self.L_m]
        self.history_Z = [self.Z]
        self.history_carbon_dividend = [np.nan]
    
    #####################################################################################
    #NESTED CES

    def calc_Omega_m(self):
        term_1 = (self.prices_high_carbon_instant*self.low_carbon_preferences)
        term_2 = (self.prices_low_carbon_m*(1- self.low_carbon_preferences))
        omega_vector = (term_1/term_2)**(self.low_carbon_substitutability_array)
        return omega_vector
    
    def calc_n_tilde_m(self):
        n_tilde_m = (self.low_carbon_preferences*(self.Omega_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array))+(1-self.low_carbon_preferences))**(self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))
        return n_tilde_m
        
    
    def calc_chi_m_nested_CES(self):
        chi_m = (self.sector_preferences*(self.n_tilde_m**((self.sector_substitutability-1)/self.sector_substitutability)))/self.prices_high_carbon_instant
        return chi_m
    
    def calc_Z(self):
        common_vector_denominator = self.Omega_m*self.prices_low_carbon_m + self.prices_high_carbon_instant
        chi_pow = self.chi_m**self.sector_substitutability
        
        Z = np.matmul(chi_pow, common_vector_denominator)#is this correct[new]        
        return Z
    
    def calc_consumption_quantities_nested_CES(self):
        H_m = (self.instant_expenditure*(self.chi_m**self.sector_substitutability))/self.Z
        L_m = H_m*self.Omega_m
        return H_m, L_m
    
    def calc_consumption_ratio(self):
        #correct
        ratio = self.L_m/(self.L_m + self.H_m)
        #C =  (self.low_carbon_preferences**self.low_carbon_substitutability_array)/( (self.low_carbon_preferences)**self.low_carbon_substitutability_array + (self.Q*(1- self.low_carbon_preferences))**self.low_carbon_substitutability_array)
        return ratio
    
    def calc_outward_social_influence(self):
        if self.imitation_state == "consumption":
            outward_social_influence = self.consumption_ratio
        elif self.imitation_state == "expenditure": 
            outward_social_influence = (self.L_m*self.prices_low_carbon_m)/self.instant_expenditure
        elif self.imitation_state == "common_knowledge":
            outward_social_influence = self.prices_low_carbon_m/(self.prices_high_carbon_instant*(1/self.Omega_m**(1/self.low_carbon_substitutability_array)) + self.prices_low_carbon_m)
        else: 
            raise ValueError("Invalid imitaiton_state:common_knowledge, expenditure, consumption")

        return outward_social_influence
    
    def update_consumption(self):
        #calculate consumption
        self.Omega_m = self.calc_Omega_m()
        self.n_tilde_m = self.calc_n_tilde_m()
        self.chi_m = self.calc_chi_m_nested_CES()
        self.Z = self.calc_Z()
        self.H_m, self.L_m = self.calc_consumption_quantities_nested_CES()

        self.consumption_ratio = self.calc_consumption_ratio()
        self.outward_social_influence = self.calc_outward_social_influence()

    def update_preferences(self, social_component):
        if self.static_internal_preference_state:
            low_carbon_preferences = (1 - self.phi_array)*self.low_carbon_preferences_init + self.phi_array*social_component
        else:
            low_carbon_preferences = (1 - self.phi_array)*self.low_carbon_preferences + self.phi_array*social_component

        self.low_carbon_preferences  = np.clip(low_carbon_preferences, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)#this stops the guassian error from causing A to be too large or small thereby producing nans

    def calc_utility_nested_CES(self):
        psuedo_utility = (self.low_carbon_preferences*(self.L_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)) + (1 - self.low_carbon_preferences)*(self.H_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)))**(self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))

        if self.M == 1:
            U = psuedo_utility
        else:
            interal_components_utility = self.sector_preferences*(psuedo_utility**((self.sector_substitutability -1)/self.sector_preferences))
            sum_utility = sum(interal_components_utility)
            U = sum_utility**(self.sector_substitutability/(self.sector_substitutability-1))
        return U,psuedo_utility
    
    def calc_identity(self) -> float:
        identity = np.mean(self.low_carbon_preferences)
        return identity

    def calc_total_emissions(self):      
        return sum(self.H_m)

    def update_prices(self,carbon_price_m):
        self.prices_high_carbon_instant = self.prices_high_carbon_m + carbon_price_m
        self.Q = self.prices_low_carbon_m/self.prices_high_carbon_instant

    def save_timeseries_data_state_individual(self):
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
        self.history_omega_m.append(self.Omega_m)
        self.history_chi_m.append(self.chi_m)
        self.history_identity.append(self.identity)
        self.history_flow_carbon_emissions.append(self.flow_carbon_emissions)
        self.history_utility.append(self.utility)
        self.history_pseudo_utility.append(self.pseudo_utility)
        self.history_H_m.append(self.H_m)
        self.history_L_m.append(self.L_m)
        self.history_Z.append(self.Z)
        #self.history_expenditure.append(self.instant_expenditure)
        self.history_carbon_dividend.append(self.carbon_dividend)

    def next_step(self, t: int, social_component: npt.NDArray, carbon_price_m, carbon_dividend):

        self.t = t
        
        self.carbon_dividend = carbon_dividend
        self.instant_expenditure = self.init_expenditure + self.carbon_dividend

        #update prices
        self.update_prices(carbon_price_m)

        #update preferences 
        if self.alpha_change_state != "fixed_preferences":
            self.update_preferences(social_component)

        #update_consumption
        self.update_consumption()

        #calc_identity
        self.identity = self.calc_identity()

        #calc_emissions
        self.flow_carbon_emissions = self.calc_total_emissions()

        if self.save_timeseries_data_state:
            if self.t == self.burn_in_duration + 1:
                self.utility,self.pseudo_utility = self.calc_utility_nested_CES()
                self.set_up_time_series()
            elif (self.t % self.compression_factor_state == 0) and (self.t > self.burn_in_duration):
                self.utility,self.pseudo_utility = self.calc_utility_nested_CES()
                self.save_timeseries_data_state_individual()
