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
from sklearn.preprocessing import normalize

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
        self.alpha_change = parameters["alpha_change"]
        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]

        # time
        self.t = 0

        # network
        self.M = int(round(parameters["M"]))
        self.N = int(round(parameters["N"]))

        #price
        self.prices_low_carbon = np.asarray([1]*self.M)
        self.prices_high_carbon_array =  np.asarray([0.75]*self.M)
        #self.prices_high_carbon = self.prices_low_carbon*parameters["price_high_carbon_factor"]   #np.random.uniform(0.5,1,self.M)

        self.burn_in_duration = parameters["burn_in_duration"]

        self.carbon_price = parameters["init_carbon_price"]
        #self.redistribution_state = parameters["redistribution_state"]
        #self.dividend_progressiveness = parameters["dividend_progressiveness"]
        self.carbon_price_duration = parameters["carbon_price_duration"]
        self.carbon_price_increased = parameters["carbon_price_increased"]
        self.carbon_tax_implementation = parameters["carbon_tax_implementation"]
        
        #if  self.carbon_tax_implementation == "linear":
        #    self.carbon_price_gradient = self.carbon_price_increased/(parameters["time_steps_max"] - self.carbon_price_duration)
            #print("carbon_price_gradient", self.carbon_price_gradient)
        #self.carbon_price_gradient = self.carbon_price_increased/(parameters["time_steps_max"] - self.carbon_price_duration)
        self.carbon_price_gradient = self.carbon_price_increased/self.carbon_price_duration

        self.service_substitutability = parameters["service_substitutability"]
        self.budget_inequality_state = parameters["budget_inequality_state"]
        self.heterogenous_preferences = parameters["heterogenous_preferences"]

        # social learning and bias
        self.confirmation_bias = parameters["confirmation_bias"]
        self.learning_error_scale = parameters["learning_error_scale"]
        self.ratio_preference_or_consumption = parameters["ratio_preference_or_consumption"]
        self.clipping_epsilon = parameters["clipping_epsilon"]
        self.ratio_preference_or_consumption_identity = parameters["ratio_preference_or_consumption_identity"]

        # setting arrays with lin space
        self.phi_array = np.asarray([parameters["phi"]]*self.M)
        #self.phi_array = np.linspace(parameters["phi_lower"], parameters["phi_upper"], num=self.M)

        #print("self.low_carbon_substitutability_array", self.low_carbon_substitutability_array)
        #print("self.prices_high_carbon_array", self.prices_high_carbon_array)
        #print("self.service_preference_matrix_init", self.service_preference_matrix_init)
        

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

        if self.alpha_change == "behavioural_independence":
            self.weighting_matrix_list = [self.weighting_matrix]*self.M

        self.network_density = nx.density(self.network)


        #THIS IS THE DIFFERNCE BETWEEN INDIVIDUALS AND THE STOCHASTIC MODEL COMPONENT

        if self.heterogenous_preferences == 1:
            #self.a_low_carbon_preference = parameters["a_low_carbon_preference"]#A
            #self.b_low_carbon_preference = parameters["b_low_carbon_preference"]#A
            self.a_identity = parameters["a_identity"]#A #IN THIS BRANCH CONSISTEN BEHAVIOURS USE THIS FOR THE IDENTITY DISTRIBUTION
            self.b_identity = parameters["b_identity"]#A #IN THIS BRANCH CONSISTEN BEHAVIOURS USE THIS FOR THE IDENTITY DISTRIBUTION
            self.var_low_carbon_preference = parameters["var_low_carbon_preference"]

            (
                self.low_carbon_preference_matrix_init
            ) = self.alt_generate_init_data_preferences()
            #) = self.generate_init_data_preferences()
            #print(" self.low_carbon_preference_matrix_init", self.low_carbon_preference_matrix_init)
            #quit()
        else:
            #this is if you want same preferences for everbody
            self.low_carbon_preference_matrix_init = np.asarray([np.random.uniform(size=self.M)]*self.N)
            #np.random.shuffle(self.low_carbon_preference_matrix_init)
            #print("self.low_carbon_preference_matrix_init", self.low_carbon_preference_matrix_init)
        

        
        
        
        """
        if self.budget_inequality_state == 1:
            #Inequality in budget
            self.budget_inequality_const = parameters["budget_inequality_const"]
            self.budget_gen_min = parameters["budget_gen_min"]
            #print("self.budget_inequality_const", self.budget_inequality_const)
            #no_norm_individual_budget_array = np.random.exponential(scale=self.budget_inequality_const, size=self.N)
            u = np.linspace(self.budget_gen_min,1,self.N) #np.random.uniform(size=self.N) #NO LONGER STOCHASTIC
            #print(u,np.random.uniform(size=self.N))
            no_norm_individual_budget_array = u**(-1/self.budget_inequality_const)       
            #no_norm_individual_budget_array = np.random.exponential(scale=self.budget_inequality_const, size=self.N)
            #print("no_norm_individual_budget_array", no_norm_individual_budget_array)
            #np.exp(-parameters["individual_budget_array_lower"]*np.linspace(parameters["individual_budget_array_lower"], parameters["individual_budget_array_upper"], num=self.N))
            self.individual_budget_array =  self.normalize_vector_sum(no_norm_individual_budget_array)
            #print("self.individual_budget_array", self.individual_budget_array,self.budget_inequality_const)
            self.gini = self.calc_gini(self.individual_budget_array)
            #print("gini", self.gini, self.budget_inequality_const)
            #quit()
        else:
            #Uniform budget
            self.individual_budget_array =  np.asarray([1/self.N]*self.N)#sums to 1

        """
        
        #Uniform budget
        self.individual_budget_array =  np.asarray([1/self.N]*self.N)#sums to 1
            
        ## LOW CARBON SUBSTITUTABLILITY - this is what defines the behaviours
        self.low_carbon_substitutability_array = np.linspace(parameters["low_carbon_substitutability_lower"], parameters["low_carbon_substitutability_upper"], num=self.M)
        #np.random.shuffle(self.low_carbon_substitutability_array)

        #HIGH CARBON PRICE
        #self.prices_high_carbon_array = np.linspace(parameters["prices_high_carbon_lower"], parameters["prices_high_carbon_upper"], num=self.M)
        #np.random.shuffle(self.prices_high_carbon_array)   
        #Uniform prices
        

        ##SERVICE PREFERENCE
        #no_norm_service_preference_matrix_init = np.linspace(parameters["service_preference_lower"], parameters["service_preference_upper"], num=self.M)
        #norm_service_preference =  self.normalize_vector_sum(no_norm_service_preference_matrix_init)
        #np.random.shuffle(norm_service_preference)
        #self.service_preference_matrix_init = np.tile(norm_service_preference, (self.N, 1)) #SO THAT IT CAN BE MADE TO BE INDIVDUAL FOR OTHER S
        
        #uniform service preferences
        self.service_preference_matrix_init =np.asarray([1/self.M]*self.M)
        

        self.agent_list = self.create_agent_list()

        self.shuffle_agent_list()#partial shuffle of the list based on identity

        if self.alpha_change == "static_preferences":
            self.social_component_matrix = np.asarray([n.low_carbon_preferences for n in self.agent_list])
            #do nothing? or feed it the same thing
        else:
            self.social_component_matrix = self.calc_social_component_matrix()

        if self.alpha_change == ("static_culturally_determined_weights" or "dynamic_culturally_determined_weights"):#update the weightings once and thats it
            self.weighting_matrix = self.update_weightings()
        elif self.alpha_change == "behavioural_independence":#independent behaviours
            self.weighting_matrix_list = self.update_weightings_list()
        

        self.init_total_carbon_emissions  = self.calc_total_emissions()
        self.total_carbon_emissions_flow = self.init_total_carbon_emissions
        #self.total_carbon_emissions_stock = self.init_total_carbon_emissions

        #if self.redistribution_state:
        #    self.carbon_dividend_array = self.calc_carbon_dividend_array()
        #else:
            #self.carbon_dividend_array = np.asarray([0]*self.N)
        self.carbon_dividend_array = np.asarray([0]*self.N)

        (
                self.identity_list,
                self.average_identity,
                self.std_identity,
                self.var_identity,
                self.min_identity,
                self.max_identity,
        ) = self.calc_network_identity()

        self.welfare = self.calc_welfare()

        #print("TIME",self.t, self.burn_in_duration, self.carbon_price_duration)
        #print("BOOL", self.t == self.burn_in_duration)
        if self.t == self.burn_in_duration:
            self.total_carbon_emissions_stock = self.total_carbon_emissions_flow
            if self.save_timeseries_data:
                self.set_up_time_series()

    def set_up_time_series(self):
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
        self.history_stock_carbon_emissions = [self.total_carbon_emissions_stock]
        self.history_flow_carbon_emissions = [self.total_carbon_emissions_flow]
        self.history_identity_list = [self.identity_list]
        self.history_welfare = [self.welfare]
        if self.budget_inequality_state == 1:
            self.history_gini = [self.gini]
    
    def normalize_vector_sum(self, vec):
        return vec/sum(vec)
    
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

    #define function to calculate Gini coefficient
    # take from: https://www.statology.org/gini-coefficient-python/
    def calc_gini(self,x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x)**2 * np.mean(x))

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

        #A_m 
        low_carbon_preference_list = [np.random.beta(self.a_low_carbon_preference, self.b_low_carbon_preference, size=self.M) for n in range(self.N)]
        #test = np.random.beta(self.a_low_carbon_preference, self.b_low_carbon_preference, size=(self.M,self.N))
        #print("asdasd")
        #print(low_carbon_preference_list, test)
        #quit()
        low_carbon_preference_matrix = np.asarray(low_carbon_preference_list)

        return low_carbon_preference_matrix#,individual_budget_matrix#, norm_service_preference_matrix,  low_carbon_substitutability_matrix ,prices_high_carbon_matrix
    
    def alt_generate_init_data_preferences(self) -> tuple[npt.NDArray, npt.NDArray]:


        indentities_beta = np.random.beta( self.a_identity, self.b_identity, size=self.N)

        preferences_uncapped = np.asarray([np.random.normal(identity,self.var_low_carbon_preference, size=self.M) for identity in  indentities_beta])

        low_carbon_preference_matrix = np.clip(preferences_uncapped, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)

        return low_carbon_preference_matrix#,individual_budget_matrix#, norm_service_preference_matrix,  low_carbon_substitutability_matrix ,prices_high_carbon_matrix

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
            "prices_high_carbon":self.prices_high_carbon_array,
            "clipping_epsilon" :self.clipping_epsilon,
            "ratio_preference_or_consumption_identity": self.ratio_preference_or_consumption_identity,
        }

        agent_list = [
            Individual(
                individual_params,
                self.low_carbon_preference_matrix_init[n],
                self.service_preference_matrix_init,
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

    def calc_ego_influence_degroot_independent(self) -> npt.NDArray:
 

        if self.ratio_preference_or_consumption == 1.0:
            attribute_matrix = np.asarray([n.low_carbon_preferences for n in self.agent_list])
        elif self.ratio_preference_or_consumption == 0.0:
            attribute_matrix = np.asarray([n.L_m/(n.L_m + n.H_m) for n in self.agent_list])
        elif self.ratio_preference_or_consumption > 0.0 and self.ratio_preference_or_consumption < 1.0:
            attribute_matrix = np.asarray([self.ratio_preference_or_consumption*n.low_carbon_preferences + (1 - self.ratio_preference_or_consumption)*(n.L_m/(n.L_m + n.H_m)) for n in self.agent_list])
        else:
            raise("Invalid ratio_preference_or_consumption = [0,1]", self.ratio_preference_or_consumption)

        #behavioural_attitude_matrix = np.asarray([n.attitudes for n in self.agent_list])
        neighbour_influence = np.zeros((self.N, self.M))

        for m in range(self.M):
            neighbour_influence[:, m] = np.matmul(self.weighting_matrix_list[m], attribute_matrix[:,m])
        
        return neighbour_influence
    
    def calc_ego_influence_degroot(self) -> npt.NDArray:

        if self.ratio_preference_or_consumption == 1.0:
            attribute_matrix = np.asarray([n.low_carbon_preferences for n in self.agent_list])
        elif self.ratio_preference_or_consumption == 0.0:
            attribute_matrix = np.asarray([n.L_m/(n.L_m + n.H_m) for n in self.agent_list])
        elif self.ratio_preference_or_consumption > 0.0 and self.ratio_preference_or_consumption < 1.0:
            attribute_matrix = np.asarray([self.ratio_preference_or_consumption*n.low_carbon_preferences + (1 - self.ratio_preference_or_consumption)*(n.L_m/(n.L_m + n.H_m)) for n in self.agent_list])
        else:
            raise("Invalid ratio_preference_or_consumption = [0,1]", self.ratio_preference_or_consumption)

        neighbour_influence = np.matmul(self.weighting_matrix, attribute_matrix)
        #print("neighbour_influence",neighbour_influence)
        
        
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

        if self.alpha_change == "behavioural_independence":
            ego_influence = self.calc_ego_influence_degroot_independent()
        else:
            ego_influence = self.calc_ego_influence_degroot()           
         

        social_influence = ego_influence + np.random.normal(loc=0, scale=self.learning_error_scale, size=(self.N, self.M))

        return social_influence

    def calc_weighting_matrix_attribute(self,attribute_array):

        #print("attribute array", attribute_array,attribute_array.shape, np.mean(attribute_array))

        difference_matrix = np.subtract.outer(attribute_array, attribute_array) #euclidean_distances(attribute_array,attribute_array)# i think this actually not doing anything? just squared at the moment
        #print("difference matrix", difference_matrix,difference_matrix.shape)
        #quit()

        alpha_numerator = np.exp(-np.multiply(self.confirmation_bias, np.abs(difference_matrix)))
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
    
    def update_weightings_list(self):

        weighting_matrix_list = []

        for m in range(self.M):
            if self.ratio_preference_or_consumption_identity == 1.0:
                low_carbon_preferences_list = np.array([x.low_carbon_preferences[m] for x in self.agent_list])
            elif self.ratio_preference_or_consumption_identity == 0.0:
                low_carbon_preferences_list = np.array([x.consumption_ratio[m] for x in self.agent_list])
            elif self.ratio_preference_or_consumption_identity > 0.0 and self.ratio_preference_or_consumption_identity < 1.0:
                low_carbon_preferences_list  =  np.array([self.ratio_preference_or_consumption_identity*x.low_carbon_preferences[m] + (1-self.ratio_preference_or_consumption_identity)*x.consumption_ratio for x in self.agent_list])
            else:
                raise("Invalid ratio_preference_or_consumption_identity = [0,1]", self.ratio_preference_or_consumption_identity)
    
            #low_carbon_preferences_list = np.array([x.low_carbon_preferences[m] for x in self.agent_list])
            norm_weighting_matrix = self.calc_weighting_matrix_attribute(low_carbon_preferences_list)
            weighting_matrix_list.append(norm_weighting_matrix)

        return weighting_matrix_list

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
        total_network_emissions = sum([x.flow_carbon_emissions for x in self.agent_list])
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

        identity_mean = np.mean(identity_list)
        identity_std = np.std(identity_list)
        identity_variance = np.var(identity_list)
        identity_max = max(identity_list)
        identity_min = min(identity_list)
        return (identity_list,identity_mean, identity_std, identity_variance, identity_max, identity_min)

    def calc_carbon_dividend_array(self):

        if self.t <= self.burn_in_duration:
            carbon_dividend_array = [0]*self.N
        else:
            wealth_list_B = np.asarray([x.init_budget for x in self.agent_list])
            tax_income_R = sum(sum(x.H_m*self.carbon_price) for x in self.agent_list)
            mean_wealth = np.mean(wealth_list_B)
            
            if self.dividend_progressiveness == 0:#percapita
                carbon_dividend_array =  np.asarray([tax_income_R/self.N]*self.N)
            elif self.dividend_progressiveness > 0 and self.dividend_progressiveness <= 1:#regressive
                d_max = - tax_income_R/(self.N*(np.max(wealth_list_B) - mean_wealth))#max value d can be
                #print("choisng min",self.dividend_progressiveness, d_max,min(self.dividend_progressiveness, d_max))
                div_prog_t = min(self.dividend_progressiveness, d_max)
                carbon_dividend_array = div_prog_t*(wealth_list_B - mean_wealth) + tax_income_R/self.N
            elif self.dividend_progressiveness >= -1 and self.dividend_progressiveness <0:#progressive
                #print("d_min", np.min(wealth_list_B) - mean_wealth)
                d_min = - tax_income_R/(self.N*(np.min(wealth_list_B) - mean_wealth))#most negative value d can be
                #print("choisng max",self.dividend_progressiveness, d_min, max(self.dividend_progressiveness, d_min))
                div_prog_t = max(self.dividend_progressiveness, d_min)
                carbon_dividend_array = div_prog_t*(wealth_list_B - mean_wealth) + tax_income_R/self.N
            else:
                raise("Invalid self.dividend_progressiveness d = [-1,1]")
        #print("carbon_dividend_array",carbon_dividend_array,self.dividend_progressiveness)
        return carbon_dividend_array
    
    def calc_carbon_price(self):
        if self.carbon_tax_implementation == "flat":
            carbon_price = self.carbon_price_increased
        elif self.carbon_tax_implementation == "linear":
            carbon_price = self.carbon_price + self.carbon_price_gradient
        else:
            raise("INVALID CARBON TAX IMPLEMENTATION")

        return carbon_price
    
    def calc_welfare(self):
        welfare = sum(i.utility for i in self.agent_list)
        return welfare

    def update_individuals(self):
        """
        Update Individual objects with new information regarding social interactions, prices and dividend
        """
        for i in range(self.N):
            self.agent_list[i].next_step(
                self.t, self.social_component_matrix[i], self.carbon_dividend_array[i], self.carbon_price
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
        self.history_stock_carbon_emissions.append(self.total_carbon_emissions_stock)
        self.history_flow_carbon_emissions.append(self.total_carbon_emissions_flow)
        self.history_identity_list.append(self.identity_list)
        self.history_welfare.append(self.welfare)
        if self.budget_inequality_state == 1:
            self.history_gini.append(self.gini)

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
        if self.alpha_change != "static_preferences":
            if self.alpha_change == "dynamic_culturally_determined_weights":
                self.weighting_matrix = self.update_weightings()
            elif self.alpha_change == "behavioural_independence":#independent behaviours
                self.weighting_matrix_list = self.update_weightings_list()

            self.social_component_matrix = self.calc_social_component_matrix()


        self.total_carbon_emissions_flow = self.calc_total_emissions()
        
        
        #print("self.total_carbon_emissions_flow",self.total_carbon_emissions_flow)

        #if self.redistribution_state:
        #    self.carbon_dividend_array = self.calc_carbon_dividend_array()
        #a = [x.instant_budget for x in self.agent_list]
        #self.gini = self.calc_gini(a)

        if self.t == self.burn_in_duration:
            self.total_carbon_emissions_stock = self.total_carbon_emissions_flow
        elif self.t > self.burn_in_duration:
            self.carbon_price = self.calc_carbon_price()
            self.total_carbon_emissions_stock = self.total_carbon_emissions_stock + self.total_carbon_emissions_flow

        
        if self.save_timeseries_data:
            if self.t == self.burn_in_duration:
                self.set_up_time_series()
            elif (self.t % self.compression_factor == 0) and (self.t > self.burn_in_duration):
                (
                        self.identity_list,
                        self.average_identity,
                        self.std_identity,
                        self.var_identity,
                        self.min_identity,
                        self.max_identity,
                ) = self.calc_network_identity()
                self.welfare = self.calc_welfare()
                if self.budget_inequality_state == 1:
                    self.gini = self.calc_gini([x.instant_budget for x in self.agent_list])
                self.save_timeseries_data_network()
