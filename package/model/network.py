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
from operator import attrgetter

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
        self.parameters = parameters

        #INITAL STATE OF THE SYSTEMS, WHAT ARE THE RUN CONDITIONS
        self.save_timeseries_data_state = parameters["save_timeseries_data_state"]
        self.compression_factor_state = parameters["compression_factor_state"]
        self.heterogenous_intrasector_preferences_state = parameters["heterogenous_intrasector_preferences_state"]
        self.heterogenous_carbon_price_state = parameters["heterogenous_carbon_price_state"]
        self.heterogenous_sector_substitutabilities_state = parameters["heterogenous_sector_substitutabilities_state"]
        self.heterogenous_phi_state = parameters["heterogenous_phi_state"]
        self.imperfect_learning_state = parameters["imperfect_learning_state"]
        self.ratio_preference_or_consumption_state = parameters["ratio_preference_or_consumption_state"]
        self.alpha_change_state = parameters["alpha_change_state"]
        self.vary_seed_imperfect_learning_state_or_initial_preferences_state = parameters["vary_seed_imperfect_learning_state_or_initial_preferences_state"]
        self.static_internal_preference_state = parameters["static_internal_preference_state"]
        self.network_type = parameters["network_type"]

        #seeds
        if self.vary_seed_imperfect_learning_state_or_initial_preferences_state:
            #if its 1 then very seed imperfect_learning_state
            self.init_vals_seed = parameters["init_vals_seed"] 
            self.set_seed = int(round(parameters["set_seed"]))
        else:
            #if not 1 then do vary seed initial preferences
            self.init_vals_seed = int(round(parameters["set_seed"]))
            self.set_seed = parameters["init_vals_seed"] 
        self.network_structure_seed = parameters["network_structure_seed"]
        np.random.seed(self.init_vals_seed)#For inital construction set a seed, this is the same for all runs, then later change it to set_seed
        
        # network
        self.N = int(round(parameters["N"]))
        if self.network_type == "SW":
            self.SW_network_density_input = parameters["SW_network_density"]
            self.SW_K = int(round((self.N - 1)*self.SW_network_density_input)) #reverse engineer the links per person using the density  d = 2m/n(n-1) where n is nodes and m number of edges
            self.SW_prob_rewire = parameters["SW_prob_rewire"]
            self.SBM_block_heterogenous_individuals_substitutabilities_state = 0#crude solution
        elif self.network_type == "SBM":
            self.SBM_block_heterogenous_individuals_substitutabilities_state = parameters["SBM_block_heterogenous_individuals_substitutabilities_state"]
            self.SBM_block_num = parameters["SBM_block_num"]
            self.SBM_network_density_input_intra_block = parameters["SBM_network_density_input_intra_block"]#within blocks
            self.SBM_network_density_input_inter_block = parameters["SBM_network_density_input_inter_block"]#between blocks
        elif self.network_type == "BA":
            self.BA_green_or_brown_hegemony = parameters["BA_green_or_brown_hegemony"]
            self.BA_nodes = parameters["BA_nodes"]
            self.SBM_block_heterogenous_individuals_substitutabilities_state = 0#crude solution
        
        self.M = int(round(parameters["M"]))

        # time
        self.t = 0
        self.burn_in_duration = parameters["burn_in_duration"]
        self.carbon_price_duration = parameters["carbon_price_duration"]

        #price
        self.prices_low_carbon = np.asarray([1]*self.M)
        self.prices_high_carbon =  np.asarray([1]*self.M)#start them at the same value
        self.carbon_price_m = np.asarray([0]*self.M)
        if self.heterogenous_carbon_price_state:
            #RIGHTWAY 
            self.carbon_price_increased_m = np.linspace(parameters["carbon_price_increased_lower"], parameters["carbon_price_increased_upper"], num=self.M)
        else:
            self.carbon_price_increased_m = np.linspace(parameters["carbon_price_increased_lower"], parameters["carbon_price_increased_lower"], num=self.M)

        # social learning and bias
        self.confirmation_bias = parameters["confirmation_bias"]
        if self.imperfect_learning_state:
            self.std_learning_error = parameters["std_learning_error"]
            self.clipping_epsilon = parameters["clipping_epsilon"]
        else:
            self.std_learning_error = 0
            self.clipping_epsilon = 0        
        self.clipping_epsilon_init_preference = parameters["clipping_epsilon_init_preference"]
        
        if self.heterogenous_phi_state:
            self.phi_array = np.linspace(parameters["phi_lower"], parameters["phi_upper"], num=self.M)
        else:
            self.phi_array = np.linspace(parameters["phi_lower"], parameters["phi_lower"], num=self.M)

        # network homophily
        self.homophily_state = parameters["homophily_state"]  # 0-1, if 1 then no mixing, if 0 then complete mixing

        self.shuffle_reps = int(
            round(self.N*(1 - self.homophily_state))
        )

########################################################
        #UP TO HEAR NUMPY RANDOM USED 1
###################################################
        # set preferences
        if self.heterogenous_intrasector_preferences_state == 1:
            self.a_identity = parameters["a_identity"]#A #IN THIS BRANCH CONSISTEN BEHAVIOURS USE THIS FOR THE IDENTITY DISTRIBUTION
            self.b_identity = parameters["b_identity"]#A #IN THIS BRANCH CONSISTEN BEHAVIOURS USE THIS FOR THE IDENTITY DISTRIBUTION
            self.std_low_carbon_preference = parameters["std_low_carbon_preference"]
            (
                self.low_carbon_preference_matrix_init
            ) = self.generate_init_data_preferences()
        else:
            #this is if you want same preferences for everbody
            self.low_carbon_preference_matrix_init = np.asarray([np.random.uniform(size=self.M)]*self.N)

        # create network
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
        ) = self.create_weighting_matrix()

        self.expenditure = parameters["expenditure"]
        self.individual_expenditure_array =  np.asarray([self.expenditure]*self.N)#sums to 1


        if (self.SBM_block_heterogenous_individuals_substitutabilities_state == 0) and (self.heterogenous_sector_substitutabilities_state == 1):
            #case 2 (based on the presentation, EXPLAIN THIS BETTER LATER)
            #YOU NEED TO HAVE UPPER AND LOWER BE DIFFERENT!
            ## LOW CARBON SUBSTITUTABLILITY - this is what defines the behaviours
            self.low_carbon_substitutability_array = np.linspace(parameters["low_carbon_substitutability_lower"], parameters["low_carbon_substitutability_upper"], num=self.M)
            self.low_carbon_substitutability_array_list = [self.low_carbon_substitutability_array]*self.N
        elif (self.SBM_block_heterogenous_individuals_substitutabilities_state == 1) and (self.heterogenous_sector_substitutabilities_state == 0):#fix this solution
            #case 3
            block_substitutabilities = np.linspace(parameters["low_carbon_substitutability_lower"], parameters["low_carbon_substitutability_upper"], num=self.SBM_block_num)
            low_carbon_substitutability_matrix = np.tile(block_substitutabilities[:, np.newaxis], (1, self.M))
            # Repeat each row according to the values in self.SBM_block_sizes
            self.low_carbon_substitutability_array_list = np.repeat(low_carbon_substitutability_matrix, self.SBM_block_sizes, axis=0)
        elif (self.SBM_block_heterogenous_individuals_substitutabilities_state == 1) and (self.heterogenous_sector_substitutabilities_state == 1):
            #case 4 - each block has different substitutabilities and so do the sectors,(base is the same?)
            self.SBM_sub_add_on = parameters["SBM_sub_add_on"]
            self.low_carbon_substitutability_array_list = []
            upper_val = parameters["low_carbon_substitutability_upper"]
            for i, block_num in enumerate(self.SBM_block_sizes):
                block_low_carbon_substitutability_array = np.linspace(parameters["low_carbon_substitutability_lower"], upper_val , num=self.M)
                block_subs = [block_low_carbon_substitutability_array]*block_num
                self.low_carbon_substitutability_array_list.extend(block_subs)
                upper_val += self.SBM_sub_add_on #with each block increase it a bit
        else:
            #case 1
            #NOTE THAT ITS UPPER HERE NOT LOWER AS I USUALLY WANT TO MAKE STUFF MORE SUBSTITUTABLE NOT LESS, AS I ASSUME THAT THE DIRECTION OF TECHNOLOGY IN GENERAL
            self.low_carbon_substitutability_array = np.linspace(parameters["low_carbon_substitutability_upper"], parameters["low_carbon_substitutability_upper"], num=self.M)
            self.low_carbon_substitutability_array_list = [self.low_carbon_substitutability_array]*self.N
            #self.low_carbon_substitutability_array = np.linspace(parameters["low_carbon_substitutability_upper"], parameters["low_carbon_substitutability_upper"], num=self.M)

        self.sector_substitutability = parameters["sector_substitutability"]
            
        self.sector_preferences = np.asarray([1/self.M]*self.M)
        
        self.agent_list = self.create_agent_list()

        self.shuffle_agent_list()#partial shuffle of the list based on identity

        #NOW SET SEED FOR THE IMPERFECT LEARNING
        np.random.seed(self.set_seed)

        if self.alpha_change_state == "fixed_preferences":
            self.social_component_matrix = np.asarray([n.low_carbon_preferences for n in self.agent_list])#DUMBY FEED IT ITSELF? DO I EVEN NEED TO DEFINE IT
        else:
            if self.alpha_change_state in ("uniform_network_weighting","static_culturally_determined_weights","dynamic_culturally_determined_weights", "common_knowledge_dynamic_culturally_determined_weights"):
                self.weighting_matrix = self.update_weightings()
            elif self.alpha_change_state in ("static_socially_determined_weights","dynamic_socially_determined_weights"):#independent behaviours
                self.weighting_matrix_list = self.update_weightings_list()
            self.social_component_matrix = self.calc_social_component_matrix()

        self.total_carbon_emissions_stock = 0#this are for post tax

        self.identity_list = list(map(attrgetter('identity'), self.agent_list))
        (
                self.average_identity,
                self.std_identity,
                self.var_identity,
                self.min_identity,
                self.max_identity,
        ) = self.calc_network_identity()

        self.welfare_stock = 0

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
        self.history_welfare_flow = [self.welfare_flow]
        self.history_welfare_stock = [self.welfare_stock]
        self.history_flow_carbon_emissions = [self.total_carbon_emissions_flow]
        self.history_stock_carbon_emissions = [self.total_carbon_emissions_stock]
    
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

    def adjust_normlize_matrix(self, matrix: npt.NDArray) -> npt.NDArray:
        """
        Row normalize an array, dela with isolated individuals

        Parameters
        ----------
        matrix: npt.NDArrayf
            array to be row normalized

        Returns
        -------
        norm_matrix: npt.NDArray
            row normalized array
        """
        # Find rows where all entries are 0
        zero_rows = (matrix.sum(axis=1) == 0)

        # Set these rows to 1 to avoid division by zero
        matrix[zero_rows, :] = 1

        # Row sums after setting 0 rows to 1
        row_sums = matrix.sum(axis=1)

        # Row normalize the array
        norm_matrix = matrix / row_sums[:, np.newaxis]

        # Put back the original 0 rows
        norm_matrix[zero_rows, :] = 0

        return norm_matrix

    def split_into_groups(self):
        if self.SBM_block_num <= 0:
            raise ValueError("SBM_block_num must be greater than zero.")

        base_count = self.N//self.SBM_block_num
        remainder = self.N % self.SBM_block_num

        # Distribute the remainder among the first few groups
        group_counts = [base_count + 1] * remainder + [base_count] * (self.SBM_block_num - remainder)
        return group_counts

    def create_weighting_matrix(self) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        """
        Create graph using Networkx library

        Parameters
        ----------
        None

        Returns
        -------
        adjacency_matrix: npt.NDArray[bool]
            adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """
        if self.network_type == "SW":
            G = nx.watts_strogatz_graph(n=self.N, k=self.SW_K, p=self.SW_prob_rewire, seed=self.set_seed)  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        elif self.network_type == "SBM":
            self.SBM_block_sizes = self.split_into_groups()
            num_blocks = len(self.SBM_block_sizes)
            # Create the stochastic block model, i can make it so that density between certain groups is different
            block_probs = np.full((num_blocks,num_blocks), self.SBM_network_density_input_inter_block)
            np.fill_diagonal(block_probs, self.SBM_network_density_input_intra_block)
            G = nx.stochastic_block_model(sizes=self.SBM_block_sizes, p=block_probs, seed=self.set_seed)
        elif self.network_type == "BA":
            G = nx.barabasi_albert_graph(n = self.N, m = self.BA_nodes)

        weighting_matrix = nx.to_numpy_array(G)

        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)
        
        self.network_density = nx.density(G)
        #print("Network density:", self.network_density)
        
        #quit()
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

        preferences_uncapped = np.asarray([np.random.normal(loc=identity,scale=self.std_low_carbon_preference, size=self.M) for identity in  indentities_beta])

        low_carbon_preference_matrix = np.clip(preferences_uncapped, 0 + self.clipping_epsilon_init_preference, 1- self.clipping_epsilon_init_preference)

        return low_carbon_preference_matrix

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
            "save_timeseries_data_state": self.save_timeseries_data_state,
            "phi_array": self.phi_array,
            "compression_factor_state": self.compression_factor_state,
            "init_carbon_price_m": self.carbon_price_m,
            #"low_carbon_substitutability": self.low_carbon_substitutability_array,
            "prices_low_carbon_m": self.prices_low_carbon,
            "prices_high_carbon_m":self.prices_high_carbon,
            "clipping_epsilon" :self.clipping_epsilon,
            "ratio_preference_or_consumption_state": self.ratio_preference_or_consumption_state,
            "sector_preferences" : self.sector_preferences,
            "burn_in_duration": self.burn_in_duration,
            "alpha_change_state": self.alpha_change_state,
            "static_internal_preference_state": self.static_internal_preference_state
        }

        individual_params["sector_substitutability"] = self.sector_substitutability

        agent_list = [
            Individual(
                individual_params,
                self.low_carbon_preference_matrix_init[n],
                #self.sector_preference_matrix_init,
                self.individual_expenditure_array[n],
                self.low_carbon_substitutability_array_list[n],
                n
            )
            for n in range(self.N)
        ]

        return agent_list
        
    def shuffle_agent_list(self): 
        #make list cirucalr then partial shuffle it
        self.agent_list.sort(key=lambda x: x.identity)#sorted by identity
        if (self.network_type== "BA") and (self.BA_green_or_brown_hegemony == 1):#WHY DOES IT ORDER IT THE WRONG WAY ROUND???
            self.agent_list.reverse()
        elif (self.network_type== "SW"):
            self.circular_agent_list()#agent list is now circular in terms of identity
        elif (self.network_type == "SBM"):
            pass
        self.partial_shuffle_agent_list()#partial shuffle of the list

    def calc_ego_influence_degroot_independent(self) -> npt.NDArray:
        #not sure if this stuff is correct tbh.

        attribute_matrix = np.asarray(list(map(attrgetter('outward_social_influence'), self.agent_list))) 

        #behavioural_attitude_matrix = np.asarray([n.attitudes for n in self.agent_list])
        neighbour_influence = np.zeros((self.N, self.M))

        for m in range(self.M):
            neighbour_influence[:, m] = np.matmul(self.weighting_matrix_list[m], attribute_matrix[:,m])
        
        return neighbour_influence
    
    def calc_ego_influence_degroot(self) -> npt.NDArray:

        attribute_matrix =np.asarray(list(map(attrgetter('outward_social_influence'), self.agent_list))) 


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

        if self.alpha_change_state in ("static_socially_determined_weights","dynamic_socially_determined_weights"):
            ego_influence = self.calc_ego_influence_degroot_independent()
        else:#culturally determined either static or dynamic
            ego_influence = self.calc_ego_influence_degroot()           
         
        social_influence = ego_influence + np.random.normal(loc=0, scale=self.std_learning_error, size=(self.N, self.M))

        return social_influence

    def calc_weighting_matrix_attribute(self,attribute_array):

        difference_matrix = np.subtract.outer(attribute_array, attribute_array) #euclidean_distances(attribute_array,attribute_array)# i think this actually not doing anything? just squared at the moment

        alpha_numerator = np.exp(-np.multiply(self.confirmation_bias, np.abs(difference_matrix)))

        non_diagonal_weighting_matrix = (
            self.adjacency_matrix*alpha_numerator
        )  # We want onlythose values that have network connections

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

        for m in range(self.M):
            low_carbon_preferences_list = attribute_matrix[m]
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
    
    def calc_welfare(self):
        welfare = sum(map(attrgetter('utility'), self.agent_list))
        #welfare = sum(i.utility for i in self.agent_list)
        return welfare

    def update_individuals(self):
        """
        Update Individual objects with new information regarding social interactions, prices and dividend
        """

        # Assuming you have self.agent_list as the list of objects
        ____ = list(map(
            lambda agent, scm: agent.next_step(self.t, scm, self.carbon_price_m),
            self.agent_list,
            self.social_component_matrix
        ))
        
    def switch_from_dynamic_to_fixed_preferences(self):
        self.alpha_change_state = "fixed_preferences"
        for i in range(self.N):
            self.agent_list[i].alpha_change_state = "fixed_preferences"
        

    def save_timeseries_data_state_network(self):
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
        self.history_welfare_flow.append(self.welfare_flow)
        self.history_welfare_stock.append(self.welfare_stock)

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

        if self.t == (self.burn_in_duration + 1):
            self.carbon_price_m = self.carbon_price_increased_m#turn on carbon price
        
        # execute step
        self.update_individuals()

        # update network parameters for next step
        if self.alpha_change_state != "fixed_preferences":
            if self.alpha_change_state in ("dynamic_culturally_determined_weights", "common_knowledge_dynamic_culturally_determined_weights"):
                self.weighting_matrix = self.update_weightings()
            elif self.alpha_change_state == "dynamic_socially_determined_weights":#independent behaviours
                self.weighting_matrix_list = self.update_weightings_list()
            else:
                pass #this is for "uniform_network_weighting", "static_socially_determined_weights","static_culturally_determined_weights"
            #This still updates for the case of the static weightings
            self.social_component_matrix = self.calc_social_component_matrix()

        #quit()

        #check the exact timings on these
        if self.t > self.burn_in_duration:#what to do it on the end so that its ready for the next round with the tax already there
            self.total_carbon_emissions_flow = self.calc_total_emissions()
            self.total_carbon_emissions_stock = self.total_carbon_emissions_stock + self.total_carbon_emissions_flow
            self.welfare_flow = self.calc_welfare()
            self.welfare_stock = self.welfare_stock + self.welfare_flow
            
        if self.save_timeseries_data_state:
            if self.t == self.burn_in_duration + 1:#want to create it the step after burn in is finished
                self.set_up_time_series()
            elif (self.t % self.compression_factor_state == 0) and (self.t > self.burn_in_duration):
                (
                        self.average_identity,
                        self.std_identity,
                        self.var_identity,
                        self.min_identity,
                        self.max_identity,
                ) = self.calc_network_identity()
                self.save_timeseries_data_state_network()
