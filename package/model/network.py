"""Create social network with individuals
A module that use input data to generate a social network containing individuals who each have multiple 
behaviours. The weighting of individuals within the social network is determined by the identity distance 
between neighbours. The simulation evolves over time saving data at set intervals to reduce data output.


Created: 10/10/2022
"""

# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
from package.model.individuals import Individual

# modules
class Network:

    def __init__(self, parameters: list):
        """
        Constructs all the necessary attributes for the Network object.

        Parameters
        ----------
        parameters : dict
            Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

        """

        self.set_seed = parameters["set_seed"]
        np.random.seed(self.set_seed)

        self.K = int(round(parameters["K"]))  # round due to the sampling method producing floats in the Sobol Sensitivity Analysis
        self.prob_rewire = parameters["prob_rewire"]

        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]

        # time
        self.t = 0

        # network
        self.M = int(round(parameters["M"]))
        self.N = int(round(parameters["N"]))

        #price
        self.prices_low_carbon = np.asarray([1]*self.M)
        #self.prices_high_carbon = self.prices_low_carbon*parameters["price_high_carbon_factor"]   #np.random.uniform(0.5,1,self.M)

        self.carbon_price = parameters["init_carbon_price"]
        self.rebate_progressiveness = parameters["rebate_progressiveness"]
        self.carbon_price_time = parameters["carbon_price_time"]
        self.carbon_price_increased = parameters["carbon_price_increased"]
        self.budget_multiplier = parameters["budget_multiplier"]

        self.carbon_tax_implementation = parameters["carbon_tax_implementation"]
        if  self.carbon_tax_implementation == "linear":
            self.carbon_price_gradient = self.carbon_price_increased/(parameters["time_steps_max"] - self.carbon_price_time)
            #print("carbon_price_gradient", self.carbon_price_gradient)

        self.service_substitutability = parameters["service_substitutability"]

        # social learning and bias
        self.confirmation_bias = parameters["confirmation_bias"]
        self.learning_error_scale = parameters["learning_error_scale"]
        self.ratio_preference_or_consumption = parameters["ratio_preference_or_consumption"]

        # social influence of behaviours
        self.phi_lower = parameters["phi_lower"]
        self.phi_upper = parameters["phi_upper"]
        self.phi_array = np.linspace(self.phi_lower, self.phi_upper, num=self.M)

        # network homophily
        self.homophily = parameters["homophily"]  # 0-1
        self.shuffle_reps = int(
            round(self.N*(1 - self.homophily))
        )

        # create network
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
        ) = self.create_weighting_matrix()

        self.network_density = nx.density(self.network)
        
        self.a_low_carbon_preference = parameters["a_low_carbon_preference"]
        self.b_low_carbon_preference = parameters["b_low_carbon_preference"]
        self.a_service_preference = parameters["a_service_preference"]
        self.b_service_preference = parameters["b_service_preference"]
        self.a_low_carbon_substitutability = parameters["a_low_carbon_substitutability"]
        self.b_low_carbon_substitutability = parameters["b_low_carbon_substitutability"]
        self.a_individual_budget = parameters["a_individual_budget"]
        self.b_individual_budget = parameters["b_individual_budget"]
        self.a_prices_high_carbon = parameters["a_prices_high_carbon"]
        self.b_prices_high_carbon = parameters["b_prices_high_carbon"]


        (
            self.low_carbon_preference_matrix_init,
            self.service_preference_matrix_init,
            self.individual_budget_array,
            self.low_carbon_substitutability_array,
            self.prices_high_carbon_array
        ) = self.generate_init_data_preferences()
        

        self.agent_list = self.create_agent_list()

        self.shuffle_agent_list()#partial shuffle of the list based on identity

        self.social_component_matrix = self.calc_social_component_matrix()

        self.init_total_carbon_emissions  = self.calc_total_emissions()
        self.total_carbon_emissions = self.init_total_carbon_emissions
        self.total_carbon_emissions_stock = self.init_total_carbon_emissions

        self.carbon_rebate_list = self.calc_carbon_rebate_list()

        (
                self.identity_list,
                self.average_identity,
                self.std_identity,
                self.var_identity,
                self.min_identity,
                self.max_identity,
        ) = self.calc_network_identity()

        if self.save_timeseries_data:
            self.history_weighting_matrix = [self.weighting_matrix]
            self.history_time = [self.t]
            self.weighting_matrix_convergence = 0  # there is no convergence in the first step, to deal with time issues when plotting
            self.history_weighting_matrix_convergence = [
                self.weighting_matrix_convergence
            ]
            self.history_average_identity = [self.average_identity]
            self.history_std_identity = [self.std_identity]
            self.history_var_identity = [self.var_identity]
            self.history_min_identity = [self.min_identity]
            self.history_max_identity = [self.max_identity]
            self.history_total_carbon_emissions = [self.total_carbon_emissions]

    def normlize_matrix(self, matrix: npt.NDArray) -> npt.NDArray:
        """
        Row normalize an array

        Parameters
        ----------
        matrix: npt.NDArrayf
            array to be row normalized

        Returns
        -------
        norm_matrix: npt.NDArray
            row normalized array
        """
        row_sums = matrix.sum(axis=1)
        norm_matrix = matrix / row_sums[:, np.newaxis]

        return norm_matrix

    def create_weighting_matrix(self) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        """
        Create watts-strogatz small world graph using Networkx library

        Parameters
        ----------
        None

        Returns
        -------
        weighting_matrix: npt.NDArray[bool]
            adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
        norm_weighting_matrix: npt.NDArray[float]
            an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the
            opinion of agent j as much as j does i's opinion
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """

        G = nx.watts_strogatz_graph(n=self.N, k=self.K, p=self.prob_rewire, seed=self.set_seed)

        weighting_matrix = nx.to_numpy_array(G)

        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)

        return (
            weighting_matrix,
            norm_weighting_matrix,
            G,
        )
    
    def circular_agent_list(self) -> list:
        """
        Makes an ordered list circular so that the start and end values are matched in value and value distribution is symmetric

        Parameters
        ----------
        list: list
            an ordered list e.g [1,2,3,4,5]
        Returns
        -------
        circular: list
            a circular list symmetric about its middle entry e.g [1,3,5,4,2]
        """

        first_half = self.agent_list[::2]  # take every second element in the list, even indicies
        second_half = (self.agent_list[1::2])[::-1]  # take every second element , odd indicies
        self.agent_list = first_half + second_half

    def partial_shuffle_agent_list(self) -> list:
        """
        Partially shuffle a list using Fisher Yates shuffle
        """

        for _ in range(self.shuffle_reps):
            a, b = np.random.randint(
                low=0, high=self.N, size=2
            )  # generate pair of indicies to swap
            self.agent_list[b], self.agent_list[a] = self.agent_list[a], self.agent_list[b]

    def generate_init_data_preferences(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Generate the initial values for agent behavioural attitudes and thresholds using Beta distribution

        Parameters
        ----------
        None

        Returns
        -------
        attitude_matrix: npt.NDArray
            NxM array of behavioural attitudes
        threshold_matrix: npt.NDArray
            NxM array of behavioural thresholds, represents the barriers to entry to performing a behaviour e.g the distance of a
            commute or disposable income of an individual
        """
        #A_m 
        low_carbon_preference_list = [np.random.beta(self.a_low_carbon_preference, self.b_low_carbon_preference, size=self.M)for n in range(self.N)]
        low_carbon_preference_matrix = np.asarray(low_carbon_preference_list)
        #a_m - normalized
        service_preference_list = [np.random.beta(self.a_service_preference, self.b_service_preference, size=self.M) for n in range(self.N)]
        service_preference_matrix = np.asarray(service_preference_list)#NEEDS TO ADD UP TO ONE SO NORMALIZE EACH ROW
        norm_service_preference_matrix = self.normlize_matrix(
            service_preference_matrix
        )
        #sigma_m - 1 dimentional
        low_carbon_substitutability_list = np.random.beta(self.a_low_carbon_substitutability, self.b_low_carbon_substitutability, size=self.M)#this is a single list that is used by all individuals
        low_carbon_substitutability_matrix = np.asarray(low_carbon_substitutability_list)
        #B_i - normalized, 1 dimentional
        individual_budget_array = np.random.beta(self.a_individual_budget, self.b_individual_budget, size=self.N)
        norm_individual_budget_array = individual_budget_array/ np.linalg.norm(individual_budget_array)
        individual_budget_matrix = np.asarray(norm_individual_budget_array)*self.budget_multiplier
        #P_H
        prices_high_carbon_list = np.random.beta(self.a_prices_high_carbon, self.b_prices_high_carbon, size=self.M)#this is a single list that is used by all individuals
        prices_high_carbon_matrix = np.asarray(prices_high_carbon_list)

        return low_carbon_preference_matrix, norm_service_preference_matrix, individual_budget_matrix, low_carbon_substitutability_matrix ,prices_high_carbon_matrix

    def create_agent_list(self) -> list[Individual]:
        """
        Create list of Individual objects that each have behaviours

        Parameters
        ----------
        None

        Returns
        -------
        agent_list: list[Individual]
            List of Individual objects 
        """

        individual_params = {
            "t": self.t,
            "M": self.M,
            "save_timeseries_data": self.save_timeseries_data,
            "phi_array": self.phi_array,
            "compression_factor": self.compression_factor,
            "service_substitutability": self.service_substitutability,
            "carbon_price": self.carbon_price,
            "low_carbon_substitutability": self.low_carbon_substitutability_array,
            "prices_low_carbon": self.prices_low_carbon,
            "prices_high_carbon":self.prices_high_carbon_array
        }

        agent_list = [
            Individual(
                individual_params,
                self.low_carbon_preference_matrix_init[n],
                self.service_preference_matrix_init[n],
                self.individual_budget_array[n],
                n
            )
            for n in range(self.N)
        ]

        return agent_list
        
    def shuffle_agent_list(self): 
        #make list cirucalr then partial shuffle it
        self.agent_list.sort(key=lambda x: x.identity)#sorted by identity
        self.circular_agent_list()#agent list is now circular in terms of identity
        self.partial_shuffle_agent_list()#partial shuffle of the list

    def calc_ego_influence_degroot(self) -> npt.NDArray:
        """
        Calculate the influence of neighbours using the Degroot model of weighted aggregation

        Parameters
        ----------
        None

        Returns
        -------
        neighbour_influence: npt.NDArray
            NxM array where each row represents the influence of an Individual listening to its neighbours regarding their
            behavioural attitude opinions, this influence is weighted by the weighting_matrix
        """

        if self.ratio_preference_or_consumption < 1:
            low_carbon_preferences_matrix = np.asarray([n.low_carbon_preferences for n in self.agent_list])
            #print("low_carbon_preferences_matrix", low_carbon_preferences_matrix)
            low_high_consumption_ratio_matrix = np.asarray([(n.L_m/(n.L_m + n.H_m)) for n in self.agent_list])
            #print("low_high_consumption_ratio_matrix", low_high_consumption_ratio_matrix)
            #print("self.ratio_preference_or_consumption",self.ratio_preference_or_consumption)
            #print("self.ratio_preference_or_consumption*low_carbon_preferences_matrix",self.ratio_preference_or_consumption*low_carbon_preferences_matrix)
            #print("(1 - self.ratio_preference_or_consumption)*low_high_consumption_ratio_matrix", (1 - self.ratio_preference_or_consumption)*low_high_consumption_ratio_matrix)
            #quit()
            attribute_matrix = self.ratio_preference_or_consumption*low_carbon_preferences_matrix + (1 - self.ratio_preference_or_consumption)*low_high_consumption_ratio_matrix
            #print("attribute_matrix", attribute_matrix, np.isnan(attribute_matrix))
        else:
            attribute_matrix = np.asarray([n.low_carbon_preferences for n in self.agent_list])

        neighbour_influence = np.matmul(self.weighting_matrix, attribute_matrix)
        
        return neighbour_influence

    def calc_social_component_matrix(self) -> npt.NDArray:
        """
        Combine neighbour influence and social learning error to updated individual behavioural attitudes

        Parameters
        ----------
        None

        Returns
        -------
        social_influence: npt.NDArray
            NxM array giving the influence of social learning from neighbours for that time step
        """

        ego_influence = self.calc_ego_influence_degroot()           

        social_influence = ego_influence + np.random.normal(
            loc=0, scale=self.learning_error_scale, size=(self.N, self.M)
        )
        return social_influence

    def calc_weighting_matrix_attribute(self,attribute_array):

        #print("attribute array", attribute_array,attribute_array.shape)

        difference_matrix = np.subtract.outer(attribute_array, attribute_array) #euclidean_distances(attribute_array,attribute_array)# i think this actually not doing anything? just squared at the moment
        #print("difference matrix", difference_matrix.shape)
        #quit()

        alpha_numerator = np.exp(
            -np.multiply(self.confirmation_bias, difference_matrix)
        )
        #print("alpha numerator", alpha_numerator)

        non_diagonal_weighting_matrix = (
            self.adjacency_matrix*alpha_numerator
        )  # We want onlythose values that have network connections

        #print("BEFORE NORMALIZING THE MATRIX<",non_diagonal_weighting_matrix )

        norm_weighting_matrix = self.normlize_matrix(
            non_diagonal_weighting_matrix
        )  # normalize the matrix row wise
    
        return norm_weighting_matrix

    def update_weightings(self) -> tuple[npt.NDArray, float]:
        """
        Update the link strength array according to the new agent identities

        Parameters
        ----------
        None

        Returns
        -------
        norm_weighting_matrix: npt.NDArray
            Row normalized weighting array giving the strength of inter-Individual connections due to similarity in identity
        total_identity_differences
        total_difference: float
            total element wise difference between the previous weighting arrays
        """
        identity_array = np.array([x.identity for x in self.agent_list])
        norm_weighting_matrix = self.calc_weighting_matrix_attribute(identity_array)

        return norm_weighting_matrix

    def calc_total_emissions(self) -> int:
        """
        Calculate total carbon emissions of N*M behaviours

        Parameters
        ----------
        None

        Returns
        -------
        total_network_emissions: float
            total network emissions from each Individual object
        """
        total_network_emissions = sum([x.total_carbon_emissions for x in self.agent_list])
        return total_network_emissions

    def calc_network_identity(self) -> tuple[float, float, float, float]:
        """
        Return various identity properties, such as mean, variance, min and max

        Parameters
        ----------
        None

        Returns
        -------
        identity_list: list
            list of individuals identity 
        identity_mean: float
            mean of network identity at time step t
        identity_std: float
            std of network identity at time step t
        identity_variance: float
            variance of network identity at time step t
        identity_max: float
            max of network identity at time step t
        identity_min: float
            min of network identity at time step t
        """
        identity_list = [x.identity for x in self.agent_list]

        #print("idntity list", identity_list )
        #quit()
        identity_mean = np.mean(identity_list)
        identity_std = np.std(identity_list)
        identity_variance = np.var(identity_list)
        identity_max = max(identity_list)
        identity_min = min(identity_list)
        return (identity_list,identity_mean, identity_std, identity_variance, identity_max, identity_min)

    def calc_carbon_rebate_list(self):
        wealth_list = [x.instant_budget for x in self.agent_list]
        tax_income = sum([x.H_m*self.carbon_price for x in self.agent_list])

        mean_wealth = np.mean(wealth_list)
        carbon_rebate_list = [self.rebate_progressiveness*(wealth_list[i] - mean_wealth) + tax_income/self.N for i in range(self.N)]
        return np.asarray(carbon_rebate_list)
    
    def calc_carbon_price(self):
        if self.carbon_tax_implementation == "flat":
            carbon_price = self.carbon_price_increased
        elif self.carbon_tax_implementation == "linear":
            carbon_price = self.carbon_price + self.carbon_price_gradient
        else:
            raise("INVALID CARBON TAX IMPLEMENTATION")

        return carbon_price


    def update_individuals(self):
        """
        Update Individual objects with new information regarding social interactions, prices and rebate
        """
        for i in range(self.N):
            self.agent_list[i].next_step(
                self.t, self.social_component_matrix[i], self.carbon_rebate_list[i], self.carbon_price
            )
    
    def save_timeseries_data_network(self):
        """
        Save time series data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.history_time.append(self.t)
        self.history_weighting_matrix.append(self.weighting_matrix)
        self.history_weighting_matrix_convergence.append(
            self.weighting_matrix_convergence
        )
        self.history_average_identity.append(self.average_identity)
        self.history_std_identity.append(self.std_identity)
        self.history_var_identity.append(self.var_identity)
        self.history_min_identity.append(self.min_identity)
        self.history_max_identity.append(self.max_identity)
        self.history_total_carbon_emissions.append(self.total_carbon_emissions)

    def next_step(self):
        """
        Push the simulation forwards one time step. First advance time, then update individuals with data from previous timestep
        then produce new data and finally save it.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # advance a time step
        self.t += 1

        # execute step
        self.update_individuals()

        # update network parameters for next step
        self.weighting_matrix = self.update_weightings()

        self.social_component_matrix = self.calc_social_component_matrix()
        self.total_carbon_emissions = self.calc_total_emissions()
        self.total_carbon_emissions_stock = self.total_carbon_emissions_stock + self.total_carbon_emissions
        (
                self.identity_list,
                self.average_identity,
                self.std_identity,
                self.var_identity,
                self.min_identity,
                self.max_identity,
        ) = self.calc_network_identity()

        #update carbon price
        #self.carbon_price = self.carbon_price + 

        #redistributed funds
        #calc how much was paid 
        # divide i then redistibute it accordingly

        self.carbon_rebate_list = self.calc_carbon_rebate_list()

        if (self.t % self.compression_factor == 0) and (self.save_timeseries_data):
            self.save_timeseries_data_network()

        if self.t > self.carbon_price_time:
            self.carbon_price = self.calc_carbon_price()
