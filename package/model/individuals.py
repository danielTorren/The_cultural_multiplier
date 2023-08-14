"""Define individual agent class
A module that defines "individuals" that have vectors of attitudes towards behaviours whose evolution
is determined through weighted social interactions.



Created: 10/10/2022
"""

# imports
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
# modules
class Individual:

    """
    Class to represent individuals with identities, preferences and consumption

    """

    def __init__(
        self,
        individual_params,
        low_carbon_preferences,
        budget,
        id_n,
    ):

        self.low_carbon_preferences_init = low_carbon_preferences   
        self.low_carbon_preferences = self.low_carbon_preferences_init       
        self.init_budget = budget
        self.instant_budget = self.init_budget

        self.carbon_price = individual_params["carbon_price"]

        self.M = individual_params["M"]
        self.t = individual_params["t"]
        self.save_timeseries_data = individual_params["save_timeseries_data"]
        self.compression_factor = individual_params["compression_factor"]
        self.phi_array = individual_params["phi_array"]
        self.service_preferences = individual_params["service_preferences"]
        self.low_carbon_substitutability_array = individual_params["low_carbon_substitutability"]
        self.prices_low_carbon = individual_params["prices_low_carbon"]
        self.prices_high_carbon = individual_params["prices_high_carbon"]
        self.clipping_epsilon = individual_params["clipping_epsilon"]
        self.ratio_preference_or_consumption_identity = individual_params["ratio_preference_or_consumption_identity"]
        self.ratio_preference_or_consumption = individual_params["ratio_preference_or_consumption"]
        self.burn_in_duration = individual_params["burn_in_duration"]

        self.utility_function_state = individual_params["utility_function_state"]

        if self.utility_function_state == "nested_CES":
            self.service_substitutability = individual_params["service_substitutability"]
        elif self.utility_function_state == "addilog_CES":
            self.lambda_m = individual_params["lambda_m"]
            self.init_vals_H = (self.instant_budget/self.M)*0.5 #assume initially its uniformaly spread

        self.psi_m = (self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array

        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price

        self.id = id_n

        self.Omega_m = self.calc_Omega_m()
        self.n_tilde_m = self.calc_n_tilde_m()
        
        if self.utility_function_state == "nested_CES":
            self.service_substitutability = individual_params["service_substitutability"]
            self.chi_m = self.calc_chi_m_nested_CES()
            self.H_m, self.L_m = self.calc_consumption_quantities_nested_CES()
            self.utility = self.calc_utility_nested_CES()
        elif self.utility_function_state == "addilog_CES":
            self.chi_m = self.calc_chi_m_addilog_CES()
            self.H_m, self.L_m = self.calc_consumption_quantities_addilog_CES()
            self.utility = self.calc_utility_addilog_CES()

        self.consumption_ratio = self.calc_consumption_ratio()
        self.outward_social_influence = self.ratio_preference_or_consumption*self.low_carbon_preferences + (1 - self.ratio_preference_or_consumption)*self.consumption_ratio

        self.identity = self.calc_identity()

        self.initial_carbon_emissions = self.calc_total_emissions()
        self.flow_carbon_emissions = self.initial_carbon_emissions

        #print("self.t",self.t, self.burn_in_duration)
        if self.t == self.burn_in_duration and self.save_timeseries_data:
            self.set_up_time_series()
    
    def set_up_time_series(self):
        self.history_low_carbon_preferences = [self.low_carbon_preferences]
        self.history_identity = [self.identity]
        self.history_flow_carbon_emissions = [self.flow_carbon_emissions]
        self.history_utility = [self.utility]
        self.history_H_m = [self.H_m]
        self.history_L_m = [self.L_m]

    def func_jacobian(self, x, chi_0, psi_0, lambda_0):
        term_1 = (chi_0 / self.chi_m)**(1 / (self.psi_m - self.lambda_m))
        term_2 = self.prices_high_carbon_instant + self.prices_low_carbon*self.Omega_m
        term_3 = (psi_0 - lambda_0) / (self.psi_m - self.lambda_m)

        jacobian = np.sum(term_1 * term_2 * term_3 * (x**(term_3 - 1)))

        return jacobian

    def func_to_solve(self, x, chi_0, psi_0, lambda_0):
        term_1 = (chi_0/self.chi_m)**(1/(self.psi_m - self.lambda_m))
        term_2 = self.prices_high_carbon_instant + self.prices_low_carbon*self.Omega_m

        f = np.sum(term_1*term_2*(x**((psi_0 - lambda_0)/(self.psi_m - self.lambda_m)))) - self.instant_budget

        return f

    def calc_chi_m_nested_CES(self):
        chi_m = ((self.service_preferences/self.prices_high_carbon_instant)**(self.service_substitutability))*(self.n_tilde_m**(self.service_substitutability-1))
        return chi_m
    
    def calc_chi_m_addilog_CES(self):
        chi_m = (self.service_preferences*self.n_tilde_m**(1-self.lambda_m))/self.prices_high_carbon_instant
        return chi_m
    
    def calc_n_tilde_m(self):
        n_tilde_m = (self.low_carbon_preferences*(self.Omega_m**self.psi_m)+(1-self.low_carbon_preferences))**(1/self.psi_m)
        return n_tilde_m
    
    def calc_Omega_m(self):       
        #print("self.low_carbon_substitutability_array",self.low_carbon_substitutability_array) 
        #print("self.prices_high_carbon_instant*self.low_carbon_preferences",self.prices_high_carbon_instant,self.low_carbon_preferences)
        term_1 = (self.prices_high_carbon_instant*self.low_carbon_preferences)
        term_2 = (self.prices_low_carbon*(1- self.low_carbon_preferences))
        omega_vector = (term_1/term_2)**(self.low_carbon_substitutability_array)
        return omega_vector

    def calc_other_H(self, H_0, chi_0, psi_0, lambda_0):
        H = ((chi_0/self.chi_m)**(1/(self.psi_m - self.lambda_m)))*((H_0)**((psi_0 - lambda_0)/(self.psi_m - self.lambda_m)))
        return H
    
    def calc_consumption_quantities_addilog_CES(self):
        root = least_squares(self.func_to_solve, x0=self.init_vals_H, jac=self.func_jacobian, bounds = (0, np.inf), args = (self.chi_m[0], self.psi_m[0], self.lambda_m[0]))

        H_0 = root["x"][0]

        H_m = self.calc_other_H(H_0,self.chi_m[0], self.psi_m[0], self.lambda_m[0])
        L_m = self.Omega_m*H_m

        ###NOT SURE I NEED THE LINE BELOW
        H_m_clipped = np.clip(H_m, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)
        L_m_clipped = np.clip(L_m, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)

        return H_m_clipped,L_m_clipped
    
    def calc_consumption_quantities_nested_CES(self):
        common_vector_denominator = self.Omega_m*self.prices_low_carbon + self.prices_high_carbon_instant

        H_m_denominators = np.matmul(self.chi_m, common_vector_denominator)

        H_m = self.instant_budget*self.chi_m/H_m_denominators
        L_m = H_m*self.Omega_m

        ###NOT SURE I NEED THE LINE BELOW
        H_m_clipped = np.clip(H_m, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)
        L_m_clipped = np.clip(L_m, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)
        
        return H_m_clipped,L_m_clipped
    
    def calc_utility_nested_CES(self):
        psuedo_utility = (self.low_carbon_preferences*(self.L_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)) + (1 - self.low_carbon_preferences)*(self.H_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)))**(self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))
        if self.M == 1:
            U = psuedo_utility
        else:
            U = (sum(self.service_preferences*(psuedo_utility**((self.service_substitutability -1)/self.service_preferences))))**(self.service_preferences/(self.service_preferences-1))
        
        return U
    
    def calc_utility_addilog_CES(self):
        term = (self.service_preferences * ((self.H_m * self.n_tilde_m) ** (1 - self.lambda_m))) / (1 - self.lambda_m)
        U = np.sum(term)
        return U
    
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
        low_carbon_preferences = (1 - self.phi_array)*self.low_carbon_preferences_init + self.phi_array*social_component
        #low_carbon_preferences = (1 - self.phi_array)*self.low_carbon_preferences + self.phi_array*social_component
        #print("prefences",self.id, self.low_carbon_preferences_init,self.low_carbon_preferences)
        ###NOT SURE I NEED THE LINE BELOW
        self.low_carbon_preferences  = np.clip(low_carbon_preferences, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)#this stops the guassian error from causing A to be too large or small thereby producing nans
        #self.low_carbon_preferences = low_carbon_preferences

    def calc_total_emissions(self):      
        return sum(self.H_m)
    
    def calc_consumption_ratio(self):
        return self.L_m/(self.L_m + self.H_m)
    
    def calc_outward_social_influence(self):
        return self.ratio_preference_or_consumption*self.low_carbon_preferences + (1 - self.ratio_preference_or_consumption)*self.consumption_ratio

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
        self.history_H_m.append(self.H_m)
        self.history_L_m.append(self.L_m)


    def next_step(self, t: int, social_component: npt.NDArray, carbon_dividend, carbon_price):

        self.t = t

        #update prices
        self.carbon_price = carbon_price
        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price
        
        #update_budget
        self.instant_budget = self.init_budget + carbon_dividend

        #update preferences 
        self.update_preferences(social_component)
        
        self.Omega_m = self.calc_Omega_m()
        self.n_tilde_m = self.calc_n_tilde_m()

        #calculate consumption
        if self.utility_function_state == "nested_CES":
            self.chi_m = self.calc_chi_m_nested_CES()
            self.H_m, self.L_m = self.calc_consumption_quantities_nested_CES()
            self.utility = self.calc_utility_nested_CES()
        elif self.utility_function_state == "addilog_CES":
            self.chi_m = self.calc_chi_m_addilog_CES()
            self.H_m, self.L_m = self.calc_consumption_quantities_addilog_CES()
            self.init_vals_H = self.H_m[0]
            self.utility = self.calc_utility_addilog_CES()

        self.consumption_ratio = self.calc_consumption_ratio()
        self.outward_social_influence = self.calc_outward_social_influence()

        #print("outward infleunce:", self.id, self.low_carbon_preferences_init, self.low_carbon_preferences, self.outward_social_influence, self.consumption_ratio)

        #calc_identity
        self.identity = self.calc_identity()

        #calc_emissions
        self.flow_carbon_emissions = self.calc_total_emissions()

        if self.save_timeseries_data:
            if self.t == self.burn_in_duration:
                self.set_up_time_series()
            elif (self.t % self.compression_factor == 0) and (self.t > self.burn_in_duration):
                self.save_timeseries_data_individual()
