# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
from operator import attrgetter
from scipy.optimize import least_squares

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
        self.set_seed = int(round(parameters["set_seed"]))
        self.network_structure_seed = parameters["network_structure_seed"]    
        self.init_vals_seed = parameters["init_vals_seed"] 

        #For inital construction set a seed, this is the same for all runs, then later change it to set_seed
        np.random.seed(self.init_vals_seed)
        
        
        # network
        self.network_density = parameters["network_density"]
        self.N = int(round(parameters["N"]))
        self.K = int(round((self.N - 1)*self.network_density)) #reverse engineer the links per person using the density  d = 2m/n(n-1) where n is nodes and m number of edges
        #print("self.K",self.K)
        self.prob_rewire = parameters["prob_rewire"]

        self.M = int(round(parameters["M"]))
        self.alpha_change = parameters["alpha_change"]
        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]
        self.utility_function_state = parameters["utility_function_state"]
        self.static_internal_A_state = parameters["static_internal_A_state"]

        # time
        self.t = 0

        #price
        self.prices_low_carbon = np.asarray([1]*self.M)
        self.prices_high_carbon_array =  np.asarray([1]*self.M)#start them at the same value

        self.burn_in_duration = parameters["burn_in_duration"]

        self.carbon_price = parameters["init_carbon_price"]
        self.redistribution_state = parameters["redistribution_state"]
        self.dividend_progressiveness = parameters["dividend_progressiveness"]
        self.carbon_price_duration = parameters["carbon_price_duration"]
        self.carbon_price_increased = parameters["carbon_price_increased"]
        self.carbon_tax_implementation = parameters["carbon_tax_implementation"]
        
        if self.carbon_tax_implementation == "linear":
            self.carbon_price_gradient = self.carbon_price_increased/self.carbon_price_duration

        if self.utility_function_state == "nested_CES":
            self.service_substitutability = parameters["service_substitutability"]
        elif self.utility_function_state == "addilog_CES":#basic and lux goods
            self.lambda_m = np.linspace(parameters["lambda_m_lower"],parameters["lambda_m_upper"],self.M)


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

        if self.heterogenous_preferences == 1:
            self.a_identity = parameters["a_identity"]#A #IN THIS BRANCH CONSISTEN BEHAVIOURS USE THIS FOR THE IDENTITY DISTRIBUTION
            self.b_identity = parameters["b_identity"]#A #IN THIS BRANCH CONSISTEN BEHAVIOURS USE THIS FOR THE IDENTITY DISTRIBUTION
            self.std_low_carbon_preference = parameters["std_low_carbon_preference"]
            (
                self.low_carbon_preference_matrix_init
            ) = self.generate_init_data_preferences()
        else:
            #this is if you want same preferences for everbody
            self.low_carbon_preference_matrix_init = np.asarray([np.random.uniform(size=self.M)]*self.N)
        
        if self.budget_inequality_state == 1:
            #Inequality in budget
            self.budget_inequality_const = parameters["budget_inequality_const"]
            u = np.linspace(0.01,1,self.N)
            no_norm_individual_budget_array = u**(-1/self.budget_inequality_const)       
            self.individual_budget_array =  self.normalize_vector_sum(no_norm_individual_budget_array)
            self.gini = self.calc_gini(self.individual_budget_array)
        else:
            #Uniform budget
            self.individual_budget_array =  np.asarray([1/self.N]*self.N)#sums to 1
            
        ## LOW CARBON SUBSTITUTABLILITY - this is what defines the behaviours
        self.low_carbon_substitutability_array = np.linspace(parameters["low_carbon_substitutability_lower"], parameters["low_carbon_substitutability_upper"], num=self.M)
        #self.low_carbon_substitutability_array = np.asarray([3])
        self.service_preferences = np.asarray([1/self.M]*self.M)
        
        self.agent_list = self.create_agent_list()

        self.shuffle_agent_list()#partial shuffle of the list based on identity

        #NOW SET SEED FOR THE IMPERFECT LEARNING
        np.random.seed(self.set_seed)

        if self.alpha_change == "static_preferences":
            self.social_component_matrix = np.asarray([n.low_carbon_preferences for n in self.agent_list])#DUMBY FEED IT ITSELF? DO I EVEN NEED TO DEFINE IT
        else:
            if self.alpha_change == ("static_culturally_determined_weights" or "dynamic_culturally_determined_weights"):
                self.weighting_matrix = self.update_weightings()
            elif self.alpha_change == "dynamic_socially_determined_weights":#independent behaviours
                self.weighting_matrix_list = self.update_weightings_list()

            self.social_component_matrix = self.calc_social_component_matrix()

        """
        if self.alpha_change == "static_preferences":
            self.social_component_matrix = np.asarray([n.low_carbon_preferences for n in self.agent_list])
            #do nothing? or feed it the same thing
        else:
            self.social_component_matrix = self.calc_social_component_matrix()

        if self.alpha_change == ("static_culturally_determined_weights" or "dynamic_culturally_determined_weights"):#update the weightings once and thats it
            self.weighting_matrix = self.update_weightings()
        elif self.alpha_change == "dynamic_socially_determined_weights":#independent behaviours
            self.weighting_matrix_list = self.update_weightings_list()

        """
        
       

        self.init_total_carbon_emissions  = self.calc_total_emissions()
        
        #self.total_carbon_emissions_stock_burn_in = self.init_total_carbon_emissions

        self.total_carbon_emissions_stock = 0#this are for post tax

        if self.redistribution_state:
            self.carbon_dividend_array = self.calc_carbon_dividend_array()
        else:
            self.carbon_dividend_array = np.asarray([0]*self.N)

        self.identity_list = list(map(attrgetter('identity'), self.agent_list))
        (
                self.average_identity,
                self.std_identity,
                self.var_identity,
                self.min_identity,
                self.max_identity,
        ) = self.calc_network_identity()

        self.welfare = self.calc_welfare()

        #if self.t == self.burn_in_duration:#in the case that we load in and there is no burn in period i want to calculate the emissions immedialty
        #    self.total_carbon_emissions_flow = self.init_total_carbon_emissions
        #    self.total_carbon_emissions_stock = self.total_carbon_emissions_flow
        #    self.set_up_time_series()

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
        self.history_identity_list = [self.identity_list]
        self.history_welfare = [self.welfare]
        self.history_flow_carbon_emissions = [self.total_carbon_emissions_flow]
        self.history_stock_carbon_emissions = [self.total_carbon_emissions_stock]

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

        G = nx.watts_strogatz_graph(n=self.N, k=self.K, p=self.prob_rewire, seed=self.network_structure_seed)#FIX THE NETWORK STRUCTURE

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

        indentities_beta = np.random.beta( self.a_identity, self.b_identity, size=self.N)

        preferences_uncapped = np.asarray([np.random.normal(identity,self.std_low_carbon_preference, size=self.M) for identity in  indentities_beta])

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
            "utility_function_state": self.utility_function_state,
            "carbon_price": self.carbon_price,
            "low_carbon_substitutability": self.low_carbon_substitutability_array,
            "prices_low_carbon": self.prices_low_carbon,
            "prices_high_carbon":self.prices_high_carbon_array,
            "clipping_epsilon" :self.clipping_epsilon,
            "ratio_preference_or_consumption_identity": self.ratio_preference_or_consumption_identity,
            "ratio_preference_or_consumption": self.ratio_preference_or_consumption,
            "service_preferences" : self.service_preferences,
            "burn_in_duration": self.burn_in_duration,
            "static_internal_A_state": self.static_internal_A_state,
            "alpha_change": self.alpha_change
        }

        if self.utility_function_state == "nested_CES":
            individual_params["service_substitutability"] = self.service_substitutability
        elif self.utility_function_state == "addilog_CES":
            individual_params["lambda_m"] = self.lambda_m

        agent_list = [
            Individual(
                individual_params,
                self.low_carbon_preference_matrix_init[n],
                #self.service_preference_matrix_init,
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
        #not sure if this stuff is corret tbh.

        attribute_matrix = np.asarray(list(map(attrgetter('outward_social_influence'), self.agent_list))) 
        """
        if self.ratio_preference_or_consumption == 1.0:
            attribute_matrix = np.asarray(list(map(attrgetter('low_carbon_preferences'), self.agent_list)))
            #attribute_matrix = np.asarray([n.low_carbon_preferences for n in self.agent_list])
        elif self.ratio_preference_or_consumption == 0.0:
            attribute_matrix = np.asarray(list(map(attrgetter('consumption_ratio'), self.agent_list)))
            #attribute_matrix = np.asarray([n.L_m/(n.L_m + n.H_m) for n in self.agent_list])
        elif self.ratio_preference_or_consumption > 0.0 and self.ratio_preference_or_consumption < 1.0:
            attribute_matrix = np.asarray([self.ratio_preference_or_consumption*n.low_carbon_preferences + (1 - self.ratio_preference_or_consumption)*(n.L_m/(n.L_m + n.H_m)) for n in self.agent_list])
        else:
            raise("Invalid ratio_preference_or_consumption = [0,1]", self.ratio_preference_or_consumption)
        """
        #behavioural_attitude_matrix = np.asarray([n.attitudes for n in self.agent_list])
        neighbour_influence = np.zeros((self.N, self.M))

        #print("attribute_matrix", attribute_matrix, attribute_matrix.shape)
        for m in range(self.M):
            #print("self.weighting_matrix_list[m]", self.weighting_matrix_list[m], (self.weighting_matrix_list[m]).shape)
            neighbour_influence[:, m] = np.matmul(self.weighting_matrix_list[m], attribute_matrix[:,m])
        
        return neighbour_influence
    
    def calc_ego_influence_degroot(self) -> npt.NDArray:

        attribute_matrix =np.asarray(list(map(attrgetter('outward_social_influence'), self.agent_list))) 
        """
        if self.ratio_preference_or_consumption == 1.0:
            attribute_matrix = np.asarray(list(map(attrgetter('low_carbon_preferences'), self.agent_list)))
            #attribute_matrix = np.asarray([n.low_carbon_preferences for n in self.agent_list])
        elif self.ratio_preference_or_consumption == 0.0:
            #attribute_matrix = np.asarray(list(map(attrgetter('low_carbon_preferences'), self.agent_list)))
            attribute_matrix = np.asarray([n.L_m/(n.L_m + n.H_m) for n in self.agent_list])
        elif self.ratio_preference_or_consumption > 0.0 and self.ratio_preference_or_consumption < 1.0:
            attribute_matrix = np.asarray([self.ratio_preference_or_consumption*n.low_carbon_preferences + (1 - self.ratio_preference_or_consumption)*(n.L_m/(n.L_m + n.H_m)) for n in self.agent_list])
        else:
            raise("Invalid ratio_preference_or_consumption = [0,1]", self.ratio_preference_or_consumption)
        """

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

        if self.alpha_change == "dynamic_socially_determined_weights":
            #print("updating socially")
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

        ##THE WEIGHTING FOR THE IDENTITY IS DONE IN INDIVIDUALS

        self.identity_list = list(map(attrgetter('identity'), self.agent_list))

        norm_weighting_matrix = self.calc_weighting_matrix_attribute(self.identity_list)

        return norm_weighting_matrix
    
    def update_weightings_list(self):

        weighting_matrix_list = []

        #take the transpose so that you can access through m, this may make it way slower
        attribute_matrix = (np.asarray(list(map(attrgetter('outward_social_influence'), self.agent_list)))).T
        #print("update_weightings_list", attribute_matrix, attribute_matrix.shape)
        for m in range(self.M):
            low_carbon_preferences_list = attribute_matrix[m]
            #print("low_carbon_preferences_list",low_carbon_preferences_list)
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

        total_network_emissions = sum(map(attrgetter('flow_carbon_emissions'), self.agent_list))
        #total_network_emissions = sum([x.flow_carbon_emissions for x in self.agent_list])
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

        #this may be slightly faster not sure
        
        #identity_list = [x.identity for x in self.agent_list]

        identity_mean = np.mean(self.identity_list)
        identity_std = np.std(self.identity_list)
        identity_variance = np.var(self.identity_list)
        identity_max = max(self.identity_list)
        identity_min = min(self.identity_list)
        return (identity_mean, identity_std, identity_variance, identity_max, identity_min)

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
        welfare = sum(map(attrgetter('utility'), self.agent_list))
        #welfare = sum(i.utility for i in self.agent_list)
        return welfare

    def update_individuals(self):
        """
        Update Individual objects with new information regarding social interactions, prices and dividend
        """
        # Assuming your list of objects is called 'object_list' and the function you want to call is 'my_function' with arguments 'arg1' and 'arg2'
        
        """
        list(map(
            partial(
                lambda agent, t, scm, cda, carbon_price: agent.next_step(t, scm, cda, carbon_price),
                t=self.t,
                scm = self.social_component_matrix,
                cda = self.carbon_dividend_array,
                carbon_price = self.carbon_price
            ),
            self.agent_list
        )
        )
        """
        
        #HAVENT BEEN ABLE TO GET THIS WORK AS I WANT IT TOO

        # Assuming you have self.agent_list as the list of objects
        ____ = list(map(
            lambda agent, scm, cda: agent.next_step_individual(self.t, scm, cda, self.carbon_price),
            self.agent_list,
            self.social_component_matrix,
            self.carbon_dividend_array
        ))
        #print("output",output)
        #quit()
        """
        for i in range(self.N):
            self.agent_list[i].next_step(
                self.t, self.social_component_matrix[i], self.carbon_dividend_array[i], self.carbon_price
            )
        """
        
    def switch_from_dynamic_to_static_preferences(self):
        self.alpha_change = "static_preferences"
        for i in range(self.N):
            self.agent_list[i].alpha_change = "static_preferences"
        

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

        if self.t > self.burn_in_duration:#what to do it on the end so that its ready for the next round with the tax already there
            self.carbon_price = self.calc_carbon_price()#update price for next round
        
        # execute step
        self.update_individuals()

        # update network parameters for next step
        if self.alpha_change != "static_preferences":
            if self.alpha_change == "dynamic_culturally_determined_weights":
                #print("updating culturally list")
                self.weighting_matrix = self.update_weightings()
            elif self.alpha_change == "dynamic_socially_determined_weights":#independent behaviours
                #print("updating socially list")
                self.weighting_matrix_list = self.update_weightings_list()

            self.social_component_matrix = self.calc_social_component_matrix()
        
        if self.redistribution_state:
            self.carbon_dividend_array = self.calc_carbon_dividend_array()
            if self.budget_inequality_state == 1:
                a = [x.instant_budget for x in self.agent_list]
                self.gini = self.calc_gini(a)

        #check the exact timings on these
        #print("self.t > self.burn_in_duration", self.t, self.burn_in_duration)
        if self.t > self.burn_in_duration:#what to do it on the end so that its ready for the next round with the tax already there
            #print("T more than burn in")
            self.total_carbon_emissions_flow = self.calc_total_emissions()
            self.total_carbon_emissions_stock = self.total_carbon_emissions_stock + self.total_carbon_emissions_flow
            #self.carbon_price = self.calc_carbon_price()#update price for next round
            
        if self.save_timeseries_data:
            if self.t == self.burn_in_duration + 1:#want to create it the step after burn in is finished
                self.set_up_time_series()
            elif (self.t % self.compression_factor == 0) and (self.t > self.burn_in_duration):
                (
                        self.average_identity,
                        self.std_identity,
                        self.var_identity,
                        self.min_identity,
                        self.max_identity,
                ) = self.calc_network_identity()
                self.welfare = self.calc_welfare()
                self.save_timeseries_data_network()

##########################################################################################################################################################
"""Define individual agent class
A module that defines "individuals" that have vectors of attitudes towards behaviours whose evolution
is determined through weighted social interactions.



Created: 10/10/2022
"""
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
        self.static_internal_A_state = individual_params["static_internal_A_state"]
        self.alpha_change = individual_params["alpha_change"]
        self.utility_function_state = individual_params["utility_function_state"]

        if self.utility_function_state == "nested_CES":
            self.service_substitutability = individual_params["service_substitutability"]
        elif self.utility_function_state == "addilog_CES":
            self.lambda_m = individual_params["lambda_m"]
            self.init_vals_H = (self.instant_budget/self.M)*0.5 #assume initially its uniformaly spread

        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price

        self.id = id_n

        self.Omega_m = self.calc_Omega_m()
       
        
        if self.utility_function_state == "nested_CES":
            #self.service_substitutability = individual_params["service_substitutability"]
            self.chi_m = self.calc_chi_m_nested_CES()
            self.H_m, self.L_m = self.calc_consumption_quantities_nested_CES()
            self.utility = self.calc_utility_nested_CES()
        elif self.utility_function_state == "addilog_CES":
            self.n_tilde_m = self.calc_n_tilde_m()
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

    #############################################
    #ADDILOG
    def func_jacobian(self, x, chi_0, psi_0, lambda_0):
        term_1 = (chi_0 / self.chi_m)**(1 / (((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array) - self.lambda_m))
        term_2 = self.prices_high_carbon_instant + self.prices_low_carbon*self.Omega_m
        term_3 = (psi_0 - lambda_0) / (((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array) - self.lambda_m)

        jacobian = np.sum(term_1 * term_2 * term_3 * (x**(term_3 - 1)))

        return jacobian

    def func_to_solve(self, x, chi_0, psi_0, lambda_0):
        term_1 = (chi_0/self.chi_m)**(1/(((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array) - self.lambda_m))
        term_2 = self.prices_high_carbon_instant + self.prices_low_carbon*self.Omega_m

        f = np.sum(term_1*term_2*(x**((psi_0 - lambda_0)/(((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array) - self.lambda_m)))) - self.instant_budget

        return f

    def calc_chi_m_addilog_CES(self):
        chi_m = (self.service_preferences*self.n_tilde_m**(1-self.lambda_m))/self.prices_high_carbon_instant
        return chi_m
    
    
    def calc_other_H(self, H_0, chi_0, psi_0, lambda_0):
        H = ((chi_0/self.chi_m)**(1/(((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array) - self.lambda_m)))*((H_0)**((psi_0 - lambda_0)/(((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array) - self.lambda_m)))
        return H
    
    def calc_consumption_quantities_addilog_CES(self):
        root = least_squares(self.func_to_solve, x0=self.init_vals_H, jac=self.func_jacobian, bounds = (0, np.inf), args = (self.chi_m[0], ((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)[0], self.lambda_m[0]))

        H_0 = root["x"][0]

        H_m = self.calc_other_H(H_0,self.chi_m[0], ((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)[0], self.lambda_m[0])
        L_m = self.Omega_m*H_m

        ###NOT SURE I NEED THE LINE BELOW
        H_m_clipped = np.clip(H_m, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)
        L_m_clipped = np.clip(L_m, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)

        return H_m_clipped,L_m_clipped
    
    def calc_utility_addilog_CES(self):
        term = (self.service_preferences * ((self.H_m * self.n_tilde_m) ** (1 - self.lambda_m))) / (1 - self.lambda_m)
        U = np.sum(term)
        return U
    
    def calc_n_tilde_m(self):
        n_tilde_m = (self.low_carbon_preferences*(self.Omega_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array))+(1-self.low_carbon_preferences))**(1/((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array))
        return n_tilde_m
    
    #####################################################################################
    #NESTED CES
    
    def calc_chi_m_nested_CES(self):
        first_term = ((self.service_preferences/self.prices_high_carbon_instant)**(self.service_substitutability))
        second_term = (self.low_carbon_preferences*(self.Omega_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)) + (1-self.low_carbon_preferences)  )**((self.service_substitutability-1)*self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))
        chi_m = first_term*second_term
        return chi_m
    
    def calc_consumption_quantities_nested_CES(self):
        common_vector_denominator = self.Omega_m*self.prices_low_carbon + self.prices_high_carbon_instant
        H_m_denominators = np.matmul(self.chi_m, common_vector_denominator)#THIS TERM IS COMMON AMONGT ALL THE Hs

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
            interal_components_utility = self.service_preferences*(psuedo_utility**((self.service_substitutability -1)/self.service_preferences))
            sum_utility = sum(interal_components_utility)
            U = sum_utility**(self.service_substitutability/(self.service_substitutability-1))
        return U
    
    ###########################################################################
    #SHARED   
    
    def calc_Omega_m(self):       
        term_1 = (self.prices_high_carbon_instant*self.low_carbon_preferences)
        term_2 = (self.prices_low_carbon*(1- self.low_carbon_preferences))
        omega_vector = (term_1/term_2)**(self.low_carbon_substitutability_array)
        return omega_vector

    def update_preferences(self, social_component):
        if self.static_internal_A_state:
            low_carbon_preferences = (1 - self.phi_array)*self.low_carbon_preferences_init + self.phi_array*social_component
        else:
            low_carbon_preferences = (1 - self.phi_array)*self.low_carbon_preferences + self.phi_array*social_component
        #
        #print("prefences",self.id, self.low_carbon_preferences_init,self.low_carbon_preferences)
        ###NOT SURE I NEED THE LINE BELOW
        self.low_carbon_preferences  = np.clip(low_carbon_preferences, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)#this stops the guassian error from causing A to be too large or small thereby producing nans
        #self.low_carbon_preferences = low_carbon_preferences

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


    def next_step_individual(self, t: int, social_component: npt.NDArray, carbon_dividend, carbon_price):

        self.t = t

        #update prices
        self.carbon_price = carbon_price
        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price
        
        #update_budget
        self.instant_budget = self.init_budget + carbon_dividend

        #update preferences 
        if self.alpha_change != "static_preferences":
            self.update_preferences(social_component)
        
        self.Omega_m = self.calc_Omega_m()
    
        #calculate consumption
        if self.utility_function_state == "nested_CES":
            self.chi_m = self.calc_chi_m_nested_CES()
            self.H_m, self.L_m = self.calc_consumption_quantities_nested_CES()
            self.utility = self.calc_utility_nested_CES()
        elif self.utility_function_state == "addilog_CES":
            self.n_tilde_m = self.calc_n_tilde_m()
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