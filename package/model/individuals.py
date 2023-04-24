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

        #print("low carb preferece", low_carbon_preferences, np.mean(low_carbon_preferences))
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

        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price

        self.id = id_n

        self.Omega_m = self.calc_omega()
        self.chi_matrix = self.calc_chi_matrix()
        self.H_m, self.L_m = self.calc_consumption_quantities()

        self.identity = self.calc_identity()
        self.initial_carbon_emissions = self.calc_total_emissions()
        self.flow_carbon_emissions = self.initial_carbon_emissions

        if self.save_timeseries_data:
            self.history_identity = [self.identity]
            self.history_flow_carbon_emissions = [self.flow_carbon_emissions]

    def calc_omega(self):
        #print("PERSON AND TIMESTEP", self.id)
        #print("self.low_carbon_preferences", self.low_carbon_preferences)
        #print("inside", (self.prices_high_carbon_instant* self.low_carbon_preferences)/(self.prices_low_carbon*(1- self.low_carbon_preferences )))
        #print("denomentator", (self.prices_low_carbon*(1- self.low_carbon_preferences )))
        #print("power", self.low_carbon_substitutability_array)
        
        omega_vector = ((self.prices_high_carbon_instant* self.low_carbon_preferences)/(self.prices_low_carbon*(1- self.low_carbon_preferences )))**(self.low_carbon_substitutability_array)
        #if np.isnan(np.sum(omega_vector)):
        #    print("self.low_carbon_preferences", self.low_carbon_preferences)
        #    print("self.prices_high_carbon_instant", self.prices_high_carbon_instant)
        #    quit()
        return omega_vector

    #I would like to make theses three functions, where the last calls the second and the second calls the first faster:
    def calc_chi(self, a, P_L, A, omega, sigma, p, m):
        
        part_one = ((a[m] * P_L[p] * A[m] * omega[p]**(1/sigma[p]))/(a[p] * P_L[m] * A[p] * omega[m]**(1/sigma[m])))**(self.service_substitutability/(1+self.service_substitutability))

        #if np.isnan(np.sum(part_one)):
        #    print("part one bits", (a[m] * P_L[p] * A[m] * omega[p]**(1/sigma[p])) ,(a[p] * P_L[m] * A[p] * omega[m]**(1/sigma[m])))
        #    print("self.low_carbon_preferences", self.low_carbon_preferences)
        #    print("self.prices_high_carbon_instant", self.prices_high_carbon_instant)
        #    quit()
        #print("part_one", part_one)
        part_two = (A[p] * omega[p]**((sigma[p]-1)/sigma[p]) + (1 - A[p]))**((sigma[p] - self.service_substitutability)/((sigma[p] - 1) * self.service_substitutability))
        #if np.isnan(np.sum(part_two)):
        #    print("part one bits",A[p] * omega[p]**((sigma[p]-1)/sigma[p]) + (1 - A[p]) ,((sigma[p] - self.service_substitutability)/((sigma[p] - 1) * self.service_substitutability)))
        #    print("self.low_carbon_preferences", self.low_carbon_preferences)
        #    print("self.prices_high_carbon_instant", self.prices_high_carbon_instant)
        #    quit()
        #print("part_two", part_two)
        part_three = (A[m] * omega[m]**((sigma[m]-1)/sigma[m]) + (1 - A[m]))**((sigma[m] - self.service_substitutability)/((sigma[m] - 1) * self.service_substitutability))
        #if np.isnan(np.sum(part_three)):
        #    print("part one bits",  (A[m] * omega[m]**((sigma[m]-1)/sigma[m]) + (1 - A[m])) ,((sigma[m] - self.service_substitutability)/((sigma[m] - 1) * self.service_substitutability)))
        #    print("self.low_carbon_preferences", self.low_carbon_preferences)
        #    print("self.prices_high_carbon_instant", self.prices_high_carbon_instant)
        #    quit()
        #print("part_three", pat_three)
        chi = part_one*part_two/part_three
        #print("chi", chi)
        return chi

    def calc_chi_matrix(self):
        chi_matrix = []
        for i in range(self.M):
            chi_row = []
            for j in range(self.M):
                value_chi = self.calc_chi(self.service_preferences, self.prices_low_carbon, self.low_carbon_preferences, self.Omega_m, self.low_carbon_substitutability_array, i, j)
                chi_row.append(value_chi)#goes p then m
            chi_matrix.append(chi_row)

        return np.asarray(chi_matrix)

    def calc_H_m_denominator(self, m):
        return sum([self.chi_matrix[p][m]*(self.Omega_m[p]*self.prices_low_carbon[p] + self.prices_high_carbon_instant[p]) for p in range(self.M)])
         

    def calc_consumption_quantities(self):
        H_m_denominators = np.asarray([self.calc_H_m_denominator(m) for m in range(self.M)])
        H_m = self.instant_budget/H_m_denominators
        L_m = H_m*self.Omega_m
        return H_m,L_m

    def calc_identity(self) -> float:
        #print("self.low_carbon_preferences", self.low_carbon_preferences)
        return np.mean(self.low_carbon_preferences)

    def update_preferences(self, social_component):
        #print("HELLO",self.low_carbon_preferences )
        low_carbon_preferences = (1 - self.phi_array)*self.low_carbon_preferences + (self.phi_array)*(social_component)
        self.low_carbon_preferences  = np.clip(low_carbon_preferences, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)#this stops the guassian error from causing A to be too large or small thereby producing nans

    def calc_total_emissions(self): 
        #print("emissions", self.H_m, sum(self.H_m))
        #quit()       
        return sum(self.H_m)

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

        self.history_identity.append(self.identity)
        self.history_flow_carbon_emissions.append(self.flow_carbon_emissions)

    def next_step(self, t: int, social_component: npt.NDArray, carbon_rebate, carbon_price):

        self.t = t

        #update prices
        self.carbon_price = carbon_price
        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price
        #update_budget
        self.instant_budget = self.init_budget + carbon_rebate
        
        #update preferences 
        self.update_preferences(social_component)
        self.identity = self.calc_identity()

        #calculate consumption
        self.Omega_m = self.calc_omega()
        self.chi_matrix = self.calc_chi_matrix()
        self.H_m, self.L_m = self.calc_consumption_quantities()

        #calc_emissions
        self.flow_carbon_emissions = self.calc_total_emissions()

        if (self.save_timeseries_data) and (self.t % self.compression_factor == 0):
            self.save_timeseries_data_individual()
