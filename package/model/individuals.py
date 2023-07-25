"""Define individual agent class
A module that defines "individuals" that have vectors of attitudes towards behaviours whose evolution
is determined through weighted social interactions.



Created: 10/10/2022
"""

# imports
import numpy as np
import numpy.typing as npt
from scipy.optimize import fsolve

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

        self.low_carbon_preferences = low_carbon_preferences

        #self.service_preferences = service_preferences

        
        
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
        
        self.service_preference = individual_params["service_preference"]
        self.lambda_1 = individual_params["lambda_2"]
        self.lambda_2 = individual_params["lambda_2"]
        init_vals_H = individual_params["init_vals_H"]

        self.burn_in_duration = individual_params["burn_in_duration"]

        self.psi_m = (self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array

        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price

        self.id = id_n

        self.Omega_m = self.calc_omega()
        self.n_tilde = self.calc_n_tilde()
        #self.chi_m = self.calc_chi_m()
        self.chi = self.calc_chi()
        self.H_m, self.L_m = self.calc_consumption_quantities(init_vals_H)
        if self.ratio_preference_or_consumption_identity < 1.0:
            self.consumption_ratio = self.calc_consumption_ratio()

        self.identity = self.calc_identity()
        self.initial_carbon_emissions = self.calc_total_emissions()
        self.flow_carbon_emissions = self.initial_carbon_emissions

        self.utility = self.calc_utility()
        
        #print("self.t",self.t, self.burn_in_duration)
        if self.t == self.burn_in_duration and self.save_timeseries_data:
            self.set_up_time_series()
    
    def set_up_time_series(self):
        self.history_low_carbon_preferences = [self.low_carbon_preferences]
        self.history_identity = [self.identity]
        self.history_flow_carbon_emissions = [self.flow_carbon_emissions]
        self.history_utility = [self.utility]
        self.history_H_1 = [self.H_m[0]]
        self.history_H_2 = [self.H_m[1]]
        self.history_L_1 = [self.L_m[0]]
        self.history_L_2 = [self.L_m[1]]


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
    
    def calc_n_tilde(self):
        n_tilde = (self.low_carbon_preferences*(self.Omega_m**self.psi_m) + (1-self.low_carbon_preferences ))**(1/self.psi_m)
        return n_tilde


    def calc_chi(self):
        chi = (((self.n_tilde[1]**(1-self.lambda_2)) * self.prices_high_carbon[1]*(1-self.service_preference))/((self.n_tilde[0]**(1-self.lambda_1))*self.prices_high_carbon[0]*self.service_preference))**(self.lambda_1/self.lambda_2)
        return chi

    def H_1_func(self,x):
        omega_term = self.prices_high_carbon + self.prices_low_carbon*self.Omega_m
        f = omega_term[1]*self.chi*x**(self.lambda_2/self.lambda_1) + omega_term[0]*x - self.instant_budget
        #print("x",x)
        return f

    def root_finder_H_1(self,init_vals):
        root = fsolve(self.H_1_func, init_vals)
        #print("ROOT",root)
        return root

    def calc_consumption_quantities(self,init_vals):

        # calc H_1
        H_1 = self.root_finder_H_1(init_vals)

        # calc H_2
        H_2 = self.chi * H_1**(self.lambda_1/self.lambda_2)
       
        # construct H_m
        H_m = np.asarray([H_1, H_2])
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
        #psuedo_utility = (self.low_carbon_preferences*(self.L_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)) + (1 - self.low_carbon_preferences)*(self.H_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)))**(self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))
        #sum_U = (sum(self.service_preferences*(psuedo_utility**((self.service_substitutability -1)/self.service_preferences))))**(self.service_preferences/(self.service_preferences-1))
        #pseudo_utility = (self.low_carbon_preferences*(self.L_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)) + (1 - self.low_carbon_preferences)*(self.H_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)))**(self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))
        
        pseudo_utility_m = self.H_m*( self.low_carbon_preferences*self.Omega_m**self.psi_m +(1-self.low_carbon_preferences))**(1/self.psi_m)
        
        U = ((1-self.service_preference)*(pseudo_utility_m[1])**(1-self.lambda_2) + self.service_preference*(1-self.lambda_2)/(1-self.lambda_1)*(pseudo_utility_m[0])**(1-self.lambda_1))**(1/(1-self.lambda_2))
        
        return U
    
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
        self.history_H_1.append(self.H_m[0])
        self.history_H_2.append(self.H_m[1])
        self.history_L_1.append(self.L_m[0])
        self.history_L_2.append(self.L_m[1])


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
        self.n_tilde = self.calc_n_tilde()
        #self.chi_m = self.calc_chi_m()
        self.chi = self.calc_chi()
        self.H_m, self.L_m = self.calc_consumption_quantities(self.H_m[0])#use last turns value as the guess
        if self.ratio_preference_or_consumption_identity < 1.0:
            self.consumption_ratio = self.calc_consumption_ratio()

        #calc_identity
        self.identity = self.calc_identity()

        #calc_emissions
        self.flow_carbon_emissions = self.calc_total_emissions()
        
        #calc_utility
        self.utility = self.calc_utility()

        if self.save_timeseries_data:
            if self.t == self.burn_in_duration:
                self.set_up_time_series()
            elif (self.t % self.compression_factor == 0) and (self.t > self.burn_in_duration):
                self.save_timeseries_data_individual()
