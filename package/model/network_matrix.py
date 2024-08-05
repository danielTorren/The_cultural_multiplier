import numpy as np
import networkx as nx
import numpy.typing as npt
from copy import deepcopy
import scipy.sparse as sp

# modules
class Network_Matrix:

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self._set_seeds()
        self._set_state_attributes()
        self._initialize_network_params()
        self._initialize_time_params()
        self._initialize_prices()
        self._initialize_social_learning()
        self._initialize_network_homophily()
        self._initialize_expenditure()
        self._initialize_sector_preferences()
        self._initialize_intra_sector_preferences()
        self._create_network()
        self._update_carbon_price()
        self.identity_vec = self._calc_identity(self.low_carbon_preference_matrix)

        if self.homophily_state != 0:
            self.low_carbon_preference_matrix = self._shuffle_preferences_start_mixed()

        self._initialize_substitutabilities()
        self._calc_consumption()
        self._initialize_social_component()
        self.carbon_dividend_array = self._calc_carbon_dividend_array()

        self.total_carbon_emissions_stock = 0
        self.total_carbon_emissions_stock_sectors = np.zeros(self.M)
        if self.network_type == "SBM":
            self.group_indices_list = self._calc_group_ids()
            self.total_carbon_emissions_stock_blocks = np.zeros(self.SBM_block_num)

    def _set_seeds(self):
        #seeds
        self.vary_seed_state = self.parameters["vary_seed_state"]
        if self.vary_seed_state =="multi":
            self.preferences_seed = self.parameters["preferences_seed"]
            self.network_structure_seed = self.parameters["network_structure_seed"]
            self.shuffle_homophily_seed = self.parameters["shuffle_homophily_seed"]
            self.shuffle_coherance_seed = self.parameters["shuffle_coherance_seed"]
        elif self.vary_seed_state =="preferences":
            self.preferences_seed = int(round(self.parameters["set_seed"]))
            self.network_structure_seed = self.parameters["network_structure_seed"]
            self.shuffle_homophily_seed = self.parameters["shuffle_homophily_seed"]
            self.shuffle_coherance_seed = self.parameters["shuffle_coherance_seed"]
        elif self.vary_seed_state =="network":
            self.preferences_seed = self.parameters["preferences_seed"]
            self.network_structure_seed = int(round(self.parameters["set_seed"]))
            self.shuffle_homophily_seed =self.parameters["shuffle_homophily_seed"]
            self.shuffle_coherance_seed = self.parameters["shuffle_coherance_seed"]
        elif self.vary_seed_state =="shuffle_homophily":
            self.preferences_seed = self.parameters["preferences_seed"]
            self.network_structure_seed = self.parameters["network_structure_seed"]
            self.shuffle_homophily_seed = int(round(self.parameters["set_seed"]))
            self.shuffle_coherance_seed = self.parameters["shuffle_coherance_seed"]
        elif self.vary_seed_state =="shuffle_coherance":
            self.preferences_seed = self.parameters["preferences_seed"]
            self.network_structure_seed = self.parameters["network_structure_seed"]
            self.shuffle_homophily_seed = self.parameters["shuffle_homophily_seed"]
            self.shuffle_coherance_seed = int(round(self.parameters["set_seed"]))


    def _set_state_attributes(self):
        self.save_timeseries_data_state = self.parameters["save_timeseries_data_state"]
        self.compression_factor_state = self.parameters["compression_factor_state"]
        self.heterogenous_intrasector_preferences_state = self.parameters["heterogenous_intrasector_preferences_state"]
        self.heterogenous_carbon_price_state = self.parameters["heterogenous_carbon_price_state"]
        self.heterogenous_sector_substitutabilities_state = self.parameters["heterogenous_sector_substitutabilities_state"]
        self.heterogenous_phi_state = self.parameters["heterogenous_phi_state"]
        self.imitation_state = self.parameters["imitation_state"]
        self.alpha_change_state = self.parameters["alpha_change_state"]
        self.vary_seed_state = self.parameters["vary_seed_state"]
        self.network_type = self.parameters["network_type"]
        """
        attr_keys = [
            "save_timeseries_data_state", "compression_factor_state", "heterogenous_intrasector_preferences_state",
            "heterogenous_carbon_price_state", "heterogenous_sector_substitutabilities_state", "heterogenous_phi_state",
            "imitation_state", "alpha_change_state", "vary_seed_state", "network_type", "N", "M", "burn_in_duration",
            "carbon_price_duration", "confirmation_bias", "clipping_epsilon_init_preference", "sector_substitutability"
        ]
        for key in attr_keys:
            setattr(self, key, self.parameters[key])
            """

    ###################################################################################################

    def _initialize_network_params(self):
        self.N = int(round(self.parameters["N"]))
        self.M = int(round(self.parameters["M"]))

        if self.network_type == "SW":
            self.SW_network_density_input = self.parameters["SW_network_density"]
            self.SW_K = int(round((self.N - 1) * self.SW_network_density_input))
            self.SW_prob_rewire = self.parameters["SW_prob_rewire"]
            self.SBM_block_heterogenous_individuals_substitutabilities_state = 0

        elif self.network_type == "SBM":
            self.SBM_block_heterogenous_individuals_substitutabilities_state = self.parameters["SBM_block_heterogenous_individuals_substitutabilities_state"]
            self.SBM_block_num = int(self.parameters["SBM_block_num"])
            self.SBM_network_density_input_intra_block = self.parameters["SBM_network_density_input_intra_block"]
            self.SBM_network_density_input_inter_block = self.parameters["SBM_network_density_input_inter_block"]

        elif self.network_type == "BA":
            self.BA_green_or_brown_hegemony = self.parameters["BA_green_or_brown_hegemony"]
            self.BA_nodes = int(self.parameters["BA_nodes"])
            self.SBM_block_heterogenous_individuals_substitutabilities_state = 0

    def _initialize_time_params(self):
        self.t = 0
        self.burn_in_duration = self.parameters["burn_in_duration"]
        self.carbon_price_duration = self.parameters["carbon_price_duration"]
        self.time_step_max = self.burn_in_duration + self.carbon_price_duration

    def _initialize_prices(self):
        self.prices_low_carbon_m = np.ones(self.M)
        self.prices_high_carbon_m = np.ones(self.M)
        self.carbon_price_m = np.zeros(self.M)

        if self.heterogenous_carbon_price_state:
            self.carbon_price_increased_m = np.linspace(self.parameters["carbon_price_increased_lower"], self.parameters["carbon_price_increased_upper"], num=self.M)
        else:
            self.carbon_price_increased_m = np.full(self.M, self.parameters["carbon_price_increased_lower"])

    def _initialize_social_learning(self):
        self.confirmation_bias = self.parameters["confirmation_bias"]
        self.clipping_epsilon_init_preference = self.parameters["clipping_epsilon_init_preference"]

        if self.heterogenous_phi_state:
            self.phi_array = np.linspace(self.parameters["phi_lower"], self.parameters["phi_upper"], num=self.M)
        else:
            self.phi_array = np.full(self.M, self.parameters["phi_lower"])

    def _initialize_network_homophily(self):
        self.homophily_state = self.parameters["homophily_state"]
        self.coherance_state = self.parameters["coherance_state"]
        self.shuffle_intensity = 1.5
        self.shuffle_reps = int(round((self.N * (1 - self.homophily_state)) ** self.shuffle_intensity))
        self.shuffle_reps_coherance = int(round((self.N * (1 - self.coherance_state)) ** self.shuffle_intensity))

    def _initialize_expenditure(self):
        self.individual_expenditure_array = np.full(self.N, 1 / self.N)
        self.instant_expenditure_vec = self.individual_expenditure_array

    def _initialize_sector_preferences(self):
        self.sector_substitutability = self.parameters["sector_substitutability"]
        self.sector_preferences = np.full(self.M, 1 / self.M)

    def _initialize_intra_sector_preferences(self):
        if self.heterogenous_intrasector_preferences_state == 1:
            self.a_preferences = self.parameters["a_preferences"]
            self.b_preferences = self.parameters["b_preferences"]
            self.std_low_carbon_preference = self.parameters["std_low_carbon_preference"]
            self.low_carbon_preference_matrix_init = self._generate_init_data_preferences_coherance()
        elif self.heterogenous_intrasector_preferences_state == 0:
            self.low_carbon_preference_matrix_init = np.full((self.N, self.M), 1 / self.M)
        else:
            self.low_carbon_preference_matrix_init = np.random.uniform(size=(self.N, self.M))

        self.low_carbon_preference_matrix = self.low_carbon_preference_matrix_init

    def _create_network(self):
        if self.network_type == "SW":
            self.network = nx.watts_strogatz_graph(n=self.N, k=self.SW_K, p=self.SW_prob_rewire, seed=self.network_structure_seed)
        elif self.network_type == "SBM":
            self.SBM_block_sizes = self._split_into_groups()
            block_probs = np.full((self.SBM_block_num, self.SBM_block_num), self.SBM_network_density_input_inter_block)
            np.fill_diagonal(block_probs, self.SBM_network_density_input_intra_block)
            self.network = nx.stochastic_block_model(sizes=self.SBM_block_sizes, p=block_probs, seed=self.network_structure_seed)
            self.block_id_list = np.asarray([i for i, size in enumerate(self.SBM_block_sizes) for _ in range(size)])
        elif self.network_type == "BA":
            self.network = nx.barabasi_albert_graph(n=self.N, m=self.BA_nodes, seed=self.network_structure_seed)

        self.adjacency_matrix = nx.to_numpy_array(self.network)
        self.sparse_adjacency_matrix = sp.csr_matrix(self.adjacency_matrix)
        # Get the non-zero indices of the adjacency matrix
        self.row_indices_sparse, self.col_indices_sparse = self.sparse_adjacency_matrix.nonzero()

        #SCIPY ALT
        #self.weighting_matrix = normalize(self.adjacency_matrix, axis=1, norm='l1')
        self.weighting_matrix = self._normlize_matrix(self.sparse_adjacency_matrix)
        self.network_density = nx.density(self.network)
    
    def _generate_init_data_preferences_coherance(self) -> tuple[npt.NDArray, npt.NDArray]:
        np.random.seed(self.preferences_seed)#For inital construction set a seed
        preferences_beta = np.random.beta( self.a_preferences, self.b_preferences, size=self.N*self.M)# THIS WILL ALWAYS PRODUCE THE SAME OUTPUT
        preferences_capped_uncoherant = np.clip(preferences_beta, 0 + self.clipping_epsilon_init_preference, 1- self.clipping_epsilon_init_preference)
        if self.coherance_state != 0:
            preferences_sorted = sorted(preferences_capped_uncoherant)#LIST IS NOW SORTED
            preferences_coherance = self._partial_shuffle_vector(np.asarray(preferences_sorted), self.shuffle_reps_coherance)
            low_carbon_preference_matrix = preferences_coherance.reshape(self.N,self.M)
        else:
            low_carbon_preference_matrix = preferences_capped_uncoherant.reshape(self.N,self.M)
        
        np.random.seed(self.shuffle_coherance_seed)#For inital construction set a seed
        np.random.shuffle(low_carbon_preference_matrix)

        return low_carbon_preference_matrix

    def _circular_list(self, list_to_circle) -> list:
        first_half = list_to_circle[::2]  # take every second element in the list, even indicies
        every_second_element = list_to_circle[1::2]# take every second element , odd indicies
        second_half = every_second_element[::-1] #reverse it
        circular_matrix = np.concatenate((first_half, second_half), axis=0)
        return circular_matrix

    def _partial_shuffle_matrix(self, matrix_to_shufle, shuffle_reps) -> list:
        self.swaps_list = []
        for _ in range(shuffle_reps):
            a, b = np.random.randint(
                low=0, high=self.N, size=2
            )  # generate pair of indicies to swap
            self.swaps_list.append((a,b))#use this to mix stuff later
            matrix_to_shufle[[a, b]] = matrix_to_shufle[[b, a]]
        return matrix_to_shufle

    def _partial_shuffle_vector(self, vector_to_shuffle, shuffle_reps) -> list:
        for _ in range(shuffle_reps):
            a, b = np.random.randint(
                low=0, high=len(vector_to_shuffle), size=2
            )
            vector_to_shuffle[a], vector_to_shuffle[b] = vector_to_shuffle[b], vector_to_shuffle[a]
        return vector_to_shuffle
    
    def _shuffle_preferences_start_mixed(self): 
        np.random.seed(self.shuffle_homophily_seed)#Set seed for shuffle
        low_carbon_preference_matrix_unsorted =  deepcopy(self.low_carbon_preference_matrix)
        identity_unsorted = np.mean(low_carbon_preference_matrix_unsorted, axis = 1)
        zipped_lists = zip(identity_unsorted, low_carbon_preference_matrix_unsorted)
        sorted_lists = sorted(zipped_lists, key=lambda x: x[0])
        _ , sorted_preferences = zip(*sorted_lists)

        if (self.network_type== "BA") and (self.BA_green_or_brown_hegemony == 1):#WHY DOES IT ORDER IT THE WRONG WAY ROUND???
            sorted_preferences = sorted_preferences[::-1]
        elif (self.network_type== "SW"):
            sorted_preferences = self._circular_list(sorted_preferences)#agent list is now circular in terms of identity
        elif (self.network_type == "SBM"):
            pass
        
        partial_shuffle_matrix = self._partial_shuffle_matrix(sorted_preferences, self.shuffle_reps)#partial shuffle of the list
        
        return partial_shuffle_matrix
    
    def _update_carbon_price(self):
        if self.t == self.burn_in_duration:
            self.carbon_price_m = self.carbon_price_increased_m
            self.prices_high_carbon_instant = self.prices_high_carbon_m + self.carbon_price_m

    def _initialize_substitutabilities(self):
        if (self.SBM_block_heterogenous_individuals_substitutabilities_state == 0) and (self.heterogenous_sector_substitutabilities_state == 1):
            self.low_carbon_substitutability_array = np.linspace(self.parameters["low_carbon_substitutability_lower"], self.parameters["low_carbon_substitutability_upper"], num=self.M)
            self.low_carbon_substitutability_matrix = np.asarray([self.low_carbon_substitutability_array]*self.N)
        elif (self.SBM_block_heterogenous_individuals_substitutabilities_state == 1) and (self.heterogenous_sector_substitutabilities_state == 0):#fix this solution
            #case 3
            block_substitutabilities = np.linspace(self.parameters["low_carbon_substitutability_lower"], self.parameters["low_carbon_substitutability_upper"], num=self.SBM_block_num)
            low_carbon_substitutability_matrix = np.tile(block_substitutabilities[:, np.newaxis], (1, self.M))
            self.low_carbon_substitutability_matrix = np.repeat(low_carbon_substitutability_matrix, self.SBM_block_sizes, axis=0)
        else:
            self.low_carbon_substitutability_array = np.linspace(self.parameters["low_carbon_substitutability_upper"], self.parameters["low_carbon_substitutability_upper"], num=self.M)
            self.low_carbon_substitutability_matrix = np.asarray([self.low_carbon_substitutability_array]*self.N)

    def _initialize_social_component(self):
        if self.alpha_change_state == "fixed_preferences":
            self.social_component_vector = self.low_carbon_preference_matrix#DUMBY FEED IT ITSELF? DO I EVEN NEED TO DEFINE IT
        else:
            if self.alpha_change_state in ("uniform_network_weighting","static_culturally_determined_weights","dynamic_identity_determined_weights", "common_knowledge_dynamic_identity_determined_weights"):
                self.weighting_matrix = self._update_weightings()
            elif self.alpha_change_state in ("static_socially_determined_weights","dynamic_socially_determined_weights"):#independent behaviours
                self.weighting_matrix_tensor = self._update_weightings_list()
            self.social_component_vector = self._calc_social_component_matrix()

    ##################################################################################
    #Below this the updating code
    ##################################################################################

    def _update_preferences(self):
        low_carbon_preferences = (1 - self.phi_array)*self.low_carbon_preference_matrix + self.phi_array*self.social_component_vector
        low_carbon_preferences  = np.clip(low_carbon_preferences, 0 + self.clipping_epsilon_init_preference, 1- self.clipping_epsilon_init_preference)#this stops the guassian error from causing A to be too large or small thereby producing nans
        return low_carbon_preferences
    
    def _calc_Omega_m(self):
        term_1 = (self.prices_high_carbon_instant*self.low_carbon_preference_matrix)
        term_2 = (self.prices_low_carbon_m*(1- self.low_carbon_preference_matrix))
        omega_vector = (term_1/term_2)**(self.low_carbon_substitutability_matrix)
        return omega_vector
    
    def _calc_n_tilde_m(self):
        n_tilde_m = (self.low_carbon_preference_matrix*(self.Omega_m_matrix**((self.low_carbon_substitutability_matrix-1)/self.low_carbon_substitutability_matrix))+(1-self.low_carbon_preference_matrix))**(self.low_carbon_substitutability_matrix/(self.low_carbon_substitutability_matrix-1))
        return n_tilde_m
        
    def _calc_chi_m_nested_CES(self):
        chi_m = ((self.sector_preferences*(self.n_tilde_m_matrix**((self.sector_substitutability-1)/self.sector_substitutability)))/self.prices_high_carbon_instant)**self.sector_substitutability
        return chi_m
    
    def _calc_Z(self):
        common_vector = self.Omega_m_matrix*self.prices_low_carbon_m + self.prices_high_carbon_instant
        no_sum_Z_terms = self.chi_m_tensor*common_vector
        Z = no_sum_Z_terms.sum(axis = 1)
        return Z
    
    def _calc_consumption_quantities_nested_CES(self):
        term_1 = self.instant_expenditure_vec/self.Z_vec
        term_1_matrix = np.tile(term_1, (self.M,1)).T
        H_m_matrix = term_1_matrix*self.chi_m_tensor
        L_m_matrix = H_m_matrix*self.Omega_m_matrix

        return H_m_matrix, L_m_matrix

    def _calc_consumption_ratio(self):
        ratio = self.L_m_matrix/(self.L_m_matrix + self.H_m_matrix)
        return ratio

    def _calc_outward_social_influence(self):
        if self.imitation_state == "consumption":
            outward_social_influence_matrix = self.consumption_ratio_matrix
        elif self.imitation_state == "expenditure": 
            outward_social_influence_matrix = (self.L_m_matrix*self.prices_low_carbon_m)/self.instant_expenditure_vec
        elif self.imitation_state == "common_knowledge":
            outward_social_influence_matrix = self.low_carbon_preference_matrix#self.prices_low_carbon_m/(self.prices_high_carbon_instant*(1/self.Omega_m_matrix**(1/self.low_carbon_substitutability_array)) + self.prices_low_carbon_m)
        else: 
            raise ValueError("Invalid imitaiton_state:common_knowledge, expenditure, consumption")
        return outward_social_influence_matrix
    
    def _calc_consumption(self):
        self.Omega_m_matrix = self._calc_Omega_m()
        self.n_tilde_m_matrix = self._calc_n_tilde_m()
        self.chi_m_tensor = self._calc_chi_m_nested_CES()
        self.Z_vec = self._calc_Z()
        self.H_m_matrix, self.L_m_matrix = self._calc_consumption_quantities_nested_CES()
        self.consumption_ratio_matrix = self._calc_consumption_ratio()
        self.outward_social_influence_matrix = self._calc_outward_social_influence()

    def _calc_ego_influence_degroot(self) -> npt.NDArray:
       
        # Perform the matrix multiplication using the dot method for sparse matrices
        neighbour_influence = self.weighting_matrix.dot(self.outward_social_influence_matrix)
        # Convert the result to a dense NumPy array if necessary
        return neighbour_influence
        
    def _calc_social_component_matrix(self) -> npt.NDArray:
        if self.alpha_change_state in ("static_socially_determined_weights","dynamic_socially_determined_weights"):
            social_influence = self._calc_ego_influence_degroot_independent()
        else:#culturally determined either static or dynamic
            social_influence = self._calc_ego_influence_degroot()           
        return social_influence

    def _calc_weighting_matrix_attribute(self, attribute_array):
        # Compute the differences only for the non-zero entries
        differences = attribute_array[self.row_indices_sparse] - attribute_array[self.col_indices_sparse]
        # Compute the weighting values only for the non-zero entries
        weights = np.exp(-self.confirmation_bias * np.abs(differences))
        # Create a sparse matrix with the same structure as the adjacency matrix
        non_diagonal_weighting_matrix = sp.csr_matrix(
            (weights, (self.row_indices_sparse, self.col_indices_sparse)),
            shape=self.adjacency_matrix.shape
        )
        # Normalize the matrix row-wise
        norm_weighting_matrix = self._normlize_matrix(non_diagonal_weighting_matrix)

        return norm_weighting_matrix

    def _normlize_matrix(self, matrix: sp.csr_matrix) -> sp.csr_matrix:
        # Normalize the matrix row-wise
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        inv_row_sums = 1.0 / row_sums
        # Create a diagonal matrix for normalization
        diagonal_matrix = sp.diags(inv_row_sums)
        # Multiply to normalize
        norm_matrix = diagonal_matrix.dot(matrix)

        return norm_matrix

    def _calc_identity(self, low_carbon_preference_matrix):
        return np.mean(low_carbon_preference_matrix, axis = 1)

    def _update_weightings(self) -> tuple[npt.NDArray, float]:
        self.identity_vec = self._calc_identity(self.low_carbon_preference_matrix)
        norm_weighting_matrix = self._calc_weighting_matrix_attribute(self.identity_vec)
        return norm_weighting_matrix

    def _calc_ego_influence_degroot_independent(self) -> npt.NDArray:
        neighbour_influence = np.zeros((self.N, self.M))
        for m in range(self.M):
            neighbour_influence[:, m] = self.weighting_matrix_tensor[m].dot(self.outward_social_influence_matrix[:, m])
            #neighbour_influence[:, m] = np.matmul(self.weighting_matrix_tensor[m], self.outward_social_influence_matrix[:,m])
        return neighbour_influence
    
    def _update_weightings_list(self) -> tuple[npt.NDArray, float]:
        weighting_matrix_list = []
        attribute_matrix = (self.outward_social_influence_matrix).T
        for m in range(self.M):
            low_carbon_preferences_list = attribute_matrix[m]
            norm_weighting_matrix = self._calc_weighting_matrix_attribute(low_carbon_preferences_list)
            weighting_matrix_list.append(norm_weighting_matrix)
        return weighting_matrix_list  

    def _calc_carbon_dividend_array(self):
        total_quantities_m = self.H_m_matrix.sum(axis = 0)
        tax_income_R =  np.sum(self.carbon_price_m*total_quantities_m) 
        carbon_dividend_array = tax_income_R/self.N
        return carbon_dividend_array

    def _calc_instant_expediture(self):
        instant_expenditure = self.individual_expenditure_array + self.carbon_dividend_array
        return instant_expenditure

    def _split_into_groups(self):
        if self.SBM_block_num <= 0:
            raise ValueError("SBM_block_num must be greater than zero.")
        base_count = self.N//self.SBM_block_num
        remainder = self.N % self.SBM_block_num
        group_counts = [base_count + 1] * remainder + [base_count] * (self.SBM_block_num - remainder)
        return group_counts

    def _calc_group_ids(self):
        group_indices_list = []
        for group_id in np.unique(self.block_id_list):
            group_indices_list.append(np.where(self.block_id_list == group_id)[0])
        return group_indices_list

    def _calc_block_emissions(self):
        block_flows = np.asarray([np.sum(self.H_m_matrix[group_indices]) for group_indices in self.group_indices_list])
        return  block_flows
    
    def _set_up_time_series(self):
        self.history_weighting_matrix = [self.weighting_matrix]
        self.history_time = [self.t]
        self.history_identity_list = [self.identity_vec]
        self.history_flow_carbon_emissions = [self.total_carbon_emissions_flow]
        self.history_stock_carbon_emissions = [self.total_carbon_emissions_stock]
        self.history_low_carbon_preference_matrix = [self.low_carbon_preference_matrix]
        self.history_identity_vec = [self.identity_vec]

    def _save_timeseries_data_state_network(self):
        self.history_time.append(self.t)
        self.history_stock_carbon_emissions.append(self.total_carbon_emissions_stock)
        self.history_flow_carbon_emissions.append(self.total_carbon_emissions_flow)
        self.history_low_carbon_preference_matrix.append(self.low_carbon_preference_matrix)
        self.history_identity_vec.append(self.identity_vec)

    def next_step(self):

        self.t += 1
        self._update_carbon_price()#check whether its time to update carbon price
        self.instant_expenditure_vec = self._calc_instant_expediture()#update expenditures with previous steps gains

        if self.alpha_change_state != "fixed_preferences":
            self.low_carbon_preference_matrix = self._update_preferences()

        self._calc_consumption()

        if self.alpha_change_state != "fixed_preferences":
            if self.alpha_change_state in ("dynamic_identity_determined_weights", "common_knowledge_dynamic_identity_determined_weights"):
                self.weighting_matrix = self._update_weightings()
            elif self.alpha_change_state == "dynamic_socially_determined_weights":#independent behaviours
                self.weighting_matrix_tensor = self._update_weightings_list()
            else:
                pass #this is for "uniform_network_weighting", "static_socially_determined_weights","static_culturally_determined_weights"
            self.social_component_vector = self._calc_social_component_matrix()

        self.carbon_dividend_array = self._calc_carbon_dividend_array()
        
        if self.t > self.burn_in_duration:#what to do it on the end so that its ready for the next round with the tax already there
            self.total_carbon_emissions_flow = self.H_m_matrix.sum()
            self.total_carbon_emissions_flow_sectors = self.H_m_matrix.sum(axis = 0)
            self.total_carbon_emissions_stock = self.total_carbon_emissions_stock + self.total_carbon_emissions_flow
            self.total_carbon_emissions_stock_sectors = self.total_carbon_emissions_stock_sectors + self.total_carbon_emissions_flow_sectors
            if self.network_type == "SBM":
                block_flows = self._calc_block_emissions()
                self.total_carbon_emissions_stock_blocks = self.total_carbon_emissions_stock_blocks + block_flows# [self.total_carbon_emissions_stock_blocks[x]+ block_flows[x] for x in range(self.SBM_block_num)]
        
        if self.save_timeseries_data_state:
            self.total_carbon_emissions_flow_vec = self.H_m_matrix.sum(axis = 1)
            if self.t == self.burn_in_duration + 1:#want to create it the step after burn in is finished
                self._set_up_time_series()
            elif (self.t % self.compression_factor_state == 0) and (self.t > self.burn_in_duration):
                self._save_timeseries_data_state_network()