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

# modules
class Network_Matrix:

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
        self.imitation_state = parameters["imitation_state"]
        self.alpha_change_state = parameters["alpha_change_state"]
        self.vary_seed_state = parameters["vary_seed_state"]
        self.static_internal_preference_state = parameters["static_internal_preference_state"]
        self.network_type = parameters["network_type"]

        #seeds
        if self.vary_seed_state =="learning":
            self.preferences_seed = parameters["preferences_seed"]
            self.network_structure_seed = parameters["network_structure_seed"]
            self.shuffle_seed = parameters["shuffle_seed"]
            self.learning_seed = int(round(parameters["set_seed"]))
        elif self.vary_seed_state =="preferences":
            self.preferences_seed = int(round(parameters["set_seed"]))
            self.network_structure_seed = parameters["network_structure_seed"]
            self.shuffle_seed = parameters["shuffle_seed"]
            self.learning_seed = parameters["learning_seed"]
        elif self.vary_seed_state =="network":
            self.preferences_seed = parameters["preferences_seed"]
            self.network_structure_seed = int(round(parameters["set_seed"]))
            self.shuffle_seed = parameters["shuffle_seed"]
            self.learning_seed = parameters["learning_seed"]
        elif self.vary_seed_state =="shuffle":
            self.preferences_seed = parameters["preferences_seed"]
            self.network_structure_seed = parameters["network_structure_seed"]
            self.shuffle_seed = int(round(parameters["set_seed"]))
            self.learning_seed = parameters["learning_seed"]
        
        # network
        self.N = int(round(parameters["N"]))
        if self.network_type == "SW":
            self.SW_network_density_input = parameters["SW_network_density"]
            self.SW_K = int(round((self.N - 1)*self.SW_network_density_input)) #reverse engineer the links per person using the density  d = 2m/n(n-1) where n is nodes and m number of edges
            self.SW_prob_rewire = parameters["SW_prob_rewire"]
            self.SBM_block_heterogenous_individuals_substitutabilities_state = 0#crude solution
        elif self.network_type == "SBM":
            self.SBM_block_heterogenous_individuals_substitutabilities_state = parameters["SBM_block_heterogenous_individuals_substitutabilities_state"]
            self.SBM_block_num = int(parameters["SBM_block_num"])
            self.SBM_network_density_input_intra_block = parameters["SBM_network_density_input_intra_block"]#within blocks
            self.SBM_network_density_input_inter_block = parameters["SBM_network_density_input_inter_block"]#between blocks
        elif self.network_type == "BA":
            self.BA_green_or_brown_hegemony = parameters["BA_green_or_brown_hegemony"]
            self.BA_nodes = int(parameters["BA_nodes"])
            self.SBM_block_heterogenous_individuals_substitutabilities_state = 0#crude solution
        
        self.M = int(round(parameters["M"]))

        # time
        self.t = 0
        self.burn_in_duration = parameters["burn_in_duration"]
        self.carbon_price_duration = parameters["carbon_price_duration"]
        self.time_step_max = self.burn_in_duration + self.carbon_price_duration

        #price
        self.prices_low_carbon_m = np.asarray([1]*self.M)
        self.prices_high_carbon_m =  np.asarray([1]*self.M)#start them at the same value
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

        #self.expenditure = parameters["expenditure"]
        self.individual_expenditure_array =  np.asarray([1/(self.N)]*self.N)#sums to 1, constant total system expenditure 
        self.instant_expenditure_vec = self.individual_expenditure_array #SET AS THE SAME INITIALLY 

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

        ########################################################
        # CANT HAVE ANYTHING USE NUMPY RANDOM BEFORE THIS
        ###################################################
        # set preferences
        np.random.seed(self.preferences_seed)#For inital construction set a seed
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
        
        self.low_carbon_preference_matrix = self.low_carbon_preference_matrix_init#THE IS THE MATRIX OF PREFERENCES, UNMIXED

        # create network
        np.random.seed(self.network_structure_seed)#This is not necessary but for consistency with other code maybe leave in
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
        ) = self.create_weighting_matrix()

        ########################################################################################################################
        # THIS STEP TAKES PLACE IN THE INDIVIDUAL USUALLY
        #########################################################################################################################
        self.update_carbon_price()#check whether its time to update carbon price

        #DO THE HOMOPHILY STUFF
        self.identity_vec = self.calc_identity()
        #print("BEFORE SHUFFLE self.identity_vec", self.identity_vec)
        np.random.seed(self.shuffle_seed)#Set seed for shuffle
        self.low_carbon_preference_matrix = self.shuffle_preferences()#partial shuffle of the list based on identity
        
        #################################################################################################################################
        #CAN NOW CALCULATE PROPERTIES OF THE AGENTS AS THEY ARE SHUFFLED CORRECTLY
        #the values if i had done it before would have been correct but the order wrong
        #I dont think i want to shuffle the adjacency matrix
    
        self.calc_consumption()#UNLIKE IN THE OTHER MODEL I CAN CALCULATE STUFF NOW
        #self.identity_vec = self.calc_identity()
        #print("MATRXI jut after update consumption 0 ",self.outward_social_influence_matrix)

        np.random.seed(self.learning_seed)#set seed for learning
        self.error_matrix_list = np.random.normal(loc=0, scale=self.std_learning_error, size=(self.time_step_max+1, self.N, self.M))
        #print("first step", self.error_matrix_list[1])

        #############################################################################################################################
        #FROM HERE MODEL IS DETERMINISTIC NOT MORE RANDOMNESS
        #############################################################################################################################
        
        # LASTLY UPDATE THE SOCIAL CONNECTIONS BASED ON START
        #CALC THE NETWORK WEIGHTING MATRX
        if self.alpha_change_state == "fixed_preferences":
            self.social_component_matrix = self.low_carbon_preference_matrix#DUMBY FEED IT ITSELF? DO I EVEN NEED TO DEFINE IT
        else:
            if self.alpha_change_state in ("uniform_network_weighting","static_culturally_determined_weights","dynamic_identity_determined_weights", "common_knowledge_dynamic_identity_determined_weights"):
                self.weighting_matrix = self.update_weightings()
            #print("self.weighting_matrix", self.weighting_matrix)
            self.social_component_matrix = self.calc_social_component_matrix()

        self.carbon_dividend_array = self.calc_carbon_dividend_array()

        #print("self.social_component_matrix", self.social_component_matrix)

        self.total_carbon_emissions_stock = 0
       
    ############################################################################################################################
    #SET UP
    ############################################################################################################################
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
            G = nx.watts_strogatz_graph(n=self.N, k=self.SW_K, p=self.SW_prob_rewire, seed=self.network_structure_seed)  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        elif self.network_type == "SBM":
            self.SBM_block_sizes = self.split_into_groups()
            num_blocks = len(self.SBM_block_sizes)
            # Create the stochastic block model, i can make it so that density between certain groups is different
            block_probs = np.full((num_blocks,num_blocks), self.SBM_network_density_input_inter_block)
            np.fill_diagonal(block_probs, self.SBM_network_density_input_intra_block)
            G = nx.stochastic_block_model(sizes=self.SBM_block_sizes, p=block_probs, seed=self.network_structure_seed)
        elif self.network_type == "BA":
            G = nx.barabasi_albert_graph(n=self.N, m=self.BA_nodes, seed= self.network_structure_seed)

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
    
    def generate_init_data_preferences(self) -> tuple[npt.NDArray, npt.NDArray]:

        indentities_beta = np.random.beta( self.a_identity, self.b_identity, size=self.N)

        preferences_uncapped = np.asarray([np.random.normal(loc=identity,scale=self.std_low_carbon_preference, size=self.M) for identity in  indentities_beta])

        low_carbon_preference_matrix = np.clip(preferences_uncapped, 0 + self.clipping_epsilon_init_preference, 1- self.clipping_epsilon_init_preference)

        return low_carbon_preference_matrix

    def circular_list(self, list_to_circle) -> list:
        first_half = list_to_circle[::2]  # take every second element in the list, even indicies
        every_second_element = list_to_circle[1::2]# take every second element , odd indicies
        second_half = every_second_element[::-1] #reverse it
        circular_matrix = np.concatenate((first_half, second_half), axis=0)
        return circular_matrix

    def partial_shuffle_matrix(self, matrix_to_shufle) -> list:
        #np.random.seed(self.shuffle_seed)#Set seed for shuffle
        #print(np.random.randint(self.shuffle_seed))
        #print("self.shuffle_seed", self.shuffle_seed)
        #print("self.shuffle_reps", self.shuffle_reps)
        #print("matrix_to_shufle", matrix_to_shufle)
        for _ in range(self.shuffle_reps):
            a, b = np.random.randint(
                low=0, high=self.N, size=2
            )  # generate pair of indicies to swap
            #print(a,b)
            #matrix_to_shufle[b], matrix_to_shufle[a] = matrix_to_shufle[a], matrix_to_shufle[b]
            matrix_to_shufle[[a, b]] = matrix_to_shufle[[b, a]]
        return matrix_to_shufle

    def shuffle_preferences(self): 
        #make list cirucalr then partial shuffle it
        sorted_indices = np.argsort(self.identity_vec)
        sorted_preferences = self.low_carbon_preference_matrix_init[sorted_indices, :]#NOW THEY ARE HOMPHILY
        
        #print("sorted_preferences no mix", sorted_preferences)

        if (self.network_type== "BA") and (self.BA_green_or_brown_hegemony == 1):#WHY DOES IT ORDER IT THE WRONG WAY ROUND???
            sorted_preferences = sorted_preferences[::-1]
        elif (self.network_type== "SW"):
            sorted_preferences = self.circular_list(sorted_preferences)#agent list is now circular in terms of identity
        elif (self.network_type == "SBM"):
            pass

        #print("CIRC", sorted_preferences)

        partial_shuffle_matrix = self.partial_shuffle_matrix(sorted_preferences)#partial shuffle of the list
        
        #print("partial_shuffle_matrix",partial_shuffle_matrix)

        return partial_shuffle_matrix
    ###########################################################################################################################################################
    #TIME LOOPS
    ###########################################################################################################################################################
    def update_carbon_price(self):
        if self.t == (self.burn_in_duration): #THIS IS REPETATIVE
            self.carbon_price_m = self.carbon_price_increased_m#turn on carbon price
            self.prices_high_carbon_instant = self.prices_high_carbon_m + self.carbon_price_m# UPDATE IT ONCE

    #CALCULATIONS FROM INDIVIDUAL
    ################################################################################################################################################
    
    
    def update_preferences(self):
        low_carbon_preferences = (1 - self.phi_array)*self.low_carbon_preference_matrix + self.phi_array*self.social_component_matrix
        low_carbon_preferences  = np.clip(low_carbon_preferences, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)#this stops the guassian error from causing A to be too large or small thereby producing nans
        return low_carbon_preferences
    
    def calc_Omega_m(self):
        term_1 = (self.prices_high_carbon_instant*self.low_carbon_preference_matrix)
        term_2 = (self.prices_low_carbon_m*(1- self.low_carbon_preference_matrix))
        omega_vector = (term_1/term_2)**(self.low_carbon_substitutability_array)
        #print("omega_vector shape", omega_vector.shape)
        #print("self.prices_high_carbon_instant", self.prices_high_carbon_instant)
        #print(self.prices_low_carbon_m)
        return omega_vector
    
    def calc_n_tilde_m(self):
        n_tilde_m = (self.low_carbon_preference_matrix*(self.Omega_m_matrix**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array))+(1-self.low_carbon_preference_matrix))**(self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))
        #print("n_tilde_m shape", n_tilde_m.shape)
        return n_tilde_m
        
    def calc_chi_m_nested_CES(self):
        chi_m = (self.sector_preferences*(self.n_tilde_m_matrix**((self.sector_substitutability-1)/self.sector_substitutability)))/self.prices_high_carbon_instant
        #print("chi_m shape",chi_m.shape)
        return chi_m
    
    def calc_Z(self):
        common_vector = self.Omega_m_matrix*self.prices_low_carbon_m + self.prices_high_carbon_instant
        chi_pow = self.chi_m_tensor**self.sector_substitutability
        no_sum_Z_terms = chi_pow*common_vector
        Z = no_sum_Z_terms.sum(axis = 1)
        return Z
    
    def calc_consumption_quantities_nested_CES(self):
        #print("MATRIX self.instant_expenditure_vec",self.t, self.instant_expenditure_vec)

        term_1 = self.instant_expenditure_vec/self.Z_vec
        #instant_expenditure_vec_matrix = np.tile(self.instant_expenditure_vec, (self.M,1)).T
        term_1_matrix = np.tile(term_1, (self.M,1)).T
        #quit()
        H_m_matrix = term_1_matrix*(self.chi_m_tensor**self.sector_substitutability)

        #H_m = (self.instant_expenditure*(self.chi_m**self.sector_substitutability))/self.Z

        L_m_matrix = H_m_matrix*self.Omega_m_matrix

        #print("H_m_matrix", H_m_matrix)

        return H_m_matrix, L_m_matrix

    def calc_consumption_ratio(self):
        ratio = self.L_m_matrix/(self.L_m_matrix + self.H_m_matrix)
        return ratio

    def calc_outward_social_influence(self):
        if self.imitation_state == "consumption":
            outward_social_influence_matrix = self.consumption_ratio_matrix
        elif self.imitation_state == "expenditure": 
            outward_social_influence_matrix = (self.L_m_matrix*self.prices_low_carbon_m)/self.instant_expenditure_vec
        elif self.imitation_state == "common_knowledge":
            outward_social_influence_matrix = self.low_carbon_preference_matrix#self.prices_low_carbon_m/(self.prices_high_carbon_instant*(1/self.Omega_m_matrix**(1/self.low_carbon_substitutability_array)) + self.prices_low_carbon_m)
        else: 
            raise ValueError("Invalid imitaiton_state:common_knowledge, expenditure, consumption")
    
        return outward_social_influence_matrix
    
    def calc_consumption(self):
        #calculate consumption
        self.Omega_m_matrix = self.calc_Omega_m()
        #print("self.Omega_m_matrix", self.Omega_m_matrix)
        self.n_tilde_m_matrix = self.calc_n_tilde_m()
        self.chi_m_tensor = self.calc_chi_m_nested_CES()
        self.Z_vec = self.calc_Z()
        self.H_m_matrix, self.L_m_matrix = self.calc_consumption_quantities_nested_CES()
        #print("self.H_m_matrix", self.H_m_matrix)
        self.consumption_ratio_matrix = self.calc_consumption_ratio()
        #print("self.consumption_ratio_matrix", self.consumption_ratio_matrix)
        self.outward_social_influence_matrix = self.calc_outward_social_influence()
        #print("self.outward_social_influence_matrix", self.outward_social_influence_matrix)
    ################################################################################################################################################
    
    def calc_ego_influence_degroot(self) -> npt.NDArray:
        #print(", self.outward_social_influence_matrix", self.outward_social_influence_matrix)
        neighbour_influence = np.matmul(self.weighting_matrix, self.outward_social_influence_matrix)

        return neighbour_influence

    def calc_social_component_matrix(self) -> npt.NDArray:

        ego_influence = self.calc_ego_influence_degroot()           
        #print("ego_influence", ego_influence)
        social_influence = ego_influence + self.error_matrix_list[self.t]

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

    def calc_identity(self):
        return np.mean(self.low_carbon_preference_matrix, axis = 1)

    def update_weightings(self) -> tuple[npt.NDArray, float]:
        self.identity_vec = self.calc_identity()
        norm_weighting_matrix = self.calc_weighting_matrix_attribute(self.identity_vec)

        return norm_weighting_matrix
    
    def calc_carbon_dividend_array(self):
        #print("self.H_m_matrix",self.t ,self.H_m_matrix)
        total_quantities_m = self.H_m_matrix.sum(axis = 0)
        #print("total_quantities_m ",total_quantities_m )
        tax_income_R =  sum(self.carbon_price_m*total_quantities_m) 
        #print("self.carbon_price_m",self.carbon_price_m)  
        #print("tax_income_R",tax_income_R) 
        carbon_dividend_array =  np.asarray([tax_income_R/self.N]*self.N)
        #print("carbon_dividend_array",carbon_dividend_array)
        return carbon_dividend_array

    def calc_instant_expediture(self):
        #print("INSIDE",self.individual_expenditure_array, self.carbon_dividend_array)
        instant_expenditure = self.individual_expenditure_array + self.carbon_dividend_array
        #print("instant_expenditure",instant_expenditure)
        return instant_expenditure
    
###############################################################
    

    def set_up_time_series(self):
        self.history_weighting_matrix = [self.weighting_matrix]
        self.history_time = [self.t]
        self.history_identity_list = [self.identity_vec]
        self.history_flow_carbon_emissions = [self.total_carbon_emissions_flow]
        self.history_stock_carbon_emissions = [self.total_carbon_emissions_stock]
        self.history_low_carbon_preference_matrix = [self.low_carbon_preference_matrix]
        self.history_identity_vec = [self.identity_vec]
        self.history_flow_carbon_emissions_vec = [self.total_carbon_emissions_flow_vec]

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
        self.history_stock_carbon_emissions.append(self.total_carbon_emissions_stock)
        self.history_flow_carbon_emissions.append(self.total_carbon_emissions_flow)
        self.history_flow_carbon_emissions_vec.append(self.total_carbon_emissions_flow_vec)
        self.history_low_carbon_preference_matrix.append(self.low_carbon_preference_matrix)
        self.history_identity_vec.append(self.identity_vec)

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

        self.update_carbon_price()#check whether its time to update carbon price
        
        self.instant_expenditure_vec = self.calc_instant_expediture()#update expenditures with previous steps gains

        #update preferences 
        if self.alpha_change_state != "fixed_preferences":
            self.low_carbon_preference_matrix = self.update_preferences()

        #update_consumption
        self.calc_consumption()

        # update network parameters for next step
        if self.alpha_change_state != "fixed_preferences":
            if self.alpha_change_state in ("dynamic_identity_determined_weights", "common_knowledge_dynamic_identity_determined_weights"):
                #print("UPDATED")
                self.weighting_matrix = self.update_weightings()
                #quit()
            else:
                pass #this is for "uniform_network_weighting", "static_socially_determined_weights","static_culturally_determined_weights"
            #This still updates for the case of the static weightings
            self.social_component_matrix = self.calc_social_component_matrix()

        self.carbon_dividend_array = self.calc_carbon_dividend_array()
        
        #check the exact timings on these
        if self.t > self.burn_in_duration:#what to do it on the end so that its ready for the next round with the tax already there
            self.total_carbon_emissions_flow = self.H_m_matrix.sum()
            self.total_carbon_emissions_stock = self.total_carbon_emissions_stock + self.total_carbon_emissions_flow

        if self.save_timeseries_data_state:
            self.total_carbon_emissions_flow_vec = self.H_m_matrix.sum(axis = 1)
            if self.t == self.burn_in_duration + 1:#want to create it the step after burn in is finished
                self.set_up_time_series()
            elif (self.t % self.compression_factor_state == 0) and (self.t > self.burn_in_duration):
                self.save_timeseries_data_state_network()