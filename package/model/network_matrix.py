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
        self._initialize_expenditure()
        self._initialize_sector_preferences()
        self._initialize_preference_coherance()
        self._initialize_intra_sector_preferences()#this generates individuals preferences with the correct level of internal coherance, but position mixed entirely
        self._update_carbon_price()
        self.identity_vec = self._calc_identity(self.low_carbon_preference_matrix)
        self._initialize_substitutabilities()
        self._calc_consumption()#Can come before network creatation as that just mixes rows of preference but not within the row ie within people

        #NETWORKS
        if self.alpha_change_state != "fixed_preferences":
            self._initialize_network_homophily()
            self._create_network()
            if self.homophily_state != 0:
                self.low_carbon_preference_matrix = self._shuffle_preferences_start_mixed()
        
        self._initialize_social_component()
        self.carbon_dividend = self._calc_carbon_dividend()

        self.total_carbon_emissions_stock = 0

    def _set_seeds(self):
        self.preferences_seed = self.parameters["preferences_seed"]
        self.network_structure_seed = self.parameters["network_structure_seed"]
        self.shuffle_homophily_seed = self.parameters["shuffle_homophily_seed"]
        self.shuffle_coherance_seed = self.parameters["shuffle_coherance_seed"]


    def _set_state_attributes(self):
        self.save_timeseries_data_state = self.parameters["save_timeseries_data_state"]
        self.compression_factor_state = self.parameters["compression_factor_state"]
        self.imitation_state = self.parameters["imitation_state"]
        self.alpha_change_state = self.parameters["alpha_change_state"]
        self.network_type = self.parameters["network_type"]
        
        self.N = int(round(self.parameters["N"]))
        self.M = int(round(self.parameters["M"]))

        self.shuffle_intensity = 1

    ###################################################################################################

    def _initialize_network_params(self):
        if self.network_type == "SW":
            self.SW_network_density_input = self.parameters["SW_network_density"]
            self.SW_K = int(round((self.N - 1) * self.SW_network_density_input))
            self.SW_prob_rewire = self.parameters["SW_prob_rewire"]

        elif self.network_type == "SBM":
            self.SBM_block_num = int(self.parameters["SBM_block_num"])
            self.SBM_network_density_input_intra_block = self.parameters["SBM_network_density_input_intra_block"]
            self.SBM_network_density_input_inter_block = self.parameters["SBM_network_density_input_inter_block"]

        elif self.network_type == "SF":
            self.SF_green_or_brown_hegemony = self.parameters["SF_green_or_brown_hegemony"]
            self.SF_density = self.parameters["SF_density"]
            self.SF_nodes = self._calculate_nodes_SF()

    def _calculate_nodes_SF(self):
        term1 = 2 * self.N - 1
        term2 = ((2 * self.N - 1) ** 2 - 4 * self.SF_density * self.N * (self.N - 1))**0.5
        nodes = (term1 - term2) / 2
        return int(round(nodes))

    def _initialize_time_params(self):
        self.t = 0
        self.burn_in_duration = self.parameters["burn_in_duration"]
        self.carbon_price_duration = self.parameters["carbon_price_duration"]
        self.time_step_max = self.burn_in_duration + self.carbon_price_duration

    def _initialize_prices(self):
        self.prices_low_carbon_m = 1
        self.prices_high_carbon_m = 1
        self.carbon_price_m = 0
        self.carbon_price_increased_m = self.parameters["carbon_price_increased"]

    def _initialize_social_learning(self):
        self.confirmation_bias = self.parameters["confirmation_bias"]
        self.clipping_epsilon_init_preference = self.parameters["clipping_epsilon_init_preference"]
        self.phi = self.parameters["phi"]

    
    def _initialize_preference_coherance(self):
        self.coherance_state = self.parameters["coherance_state"]
        self.shuffle_reps_coherance = int(round(((self.N*self.M) * (1 - self.coherance_state)) ** self.shuffle_intensity))
        #print("self.shuffle_reps_coherance", self.shuffle_reps_coherance, self.N*self.M)
    

    def _initialize_network_homophily(self):
        self.homophily_state = self.parameters["homophily_state"]
        self.shuffle_reps_homophily = int(round((self.N * (1 - self.homophily_state)) ** self.shuffle_intensity))

    def _initialize_expenditure(self):
        #self.base_expenditure_array = np.full(self.N, 1 / self.N)
        self.base_expenditure = 1/self.N#1/self.N
        #print("self.base_expenditure ", self.base_expenditure)
        self.instant_expenditure = self.base_expenditure

    def _initialize_sector_preferences(self):
        self.sector_substitutability = self.parameters["sector_substitutability"]
        self.sector_preferences = 1/ self.M

    def _initialize_intra_sector_preferences(self):
        self.a_preferences = self.parameters["a_preferences"]
        self.b_preferences = self.parameters["b_preferences"]
        self.low_carbon_preference_matrix_init = self._generate_init_data_preferences_coherance()
        self.low_carbon_preference_matrix = self.low_carbon_preference_matrix_init#have a copy of initial preferences

    def _create_network(self):
        if self.network_type == "SW":
            self.network = nx.watts_strogatz_graph(n=self.N, k=self.SW_K, p=self.SW_prob_rewire, seed=self.network_structure_seed)
        elif self.network_type == "SBM":
            self.SBM_block_sizes = self._split_into_groups()
            block_probs = np.full((self.SBM_block_num, self.SBM_block_num), self.SBM_network_density_input_inter_block)
            np.fill_diagonal(block_probs, self.SBM_network_density_input_intra_block)
            self.network = nx.stochastic_block_model(sizes=self.SBM_block_sizes, p=block_probs, seed=self.network_structure_seed)
            self.block_id_list = np.asarray([i for i, size in enumerate(self.SBM_block_sizes) for _ in range(size)])
        elif self.network_type == "SF":
            self.network = nx.barabasi_albert_graph(n=self.N, m=self.SF_nodes, seed=self.network_structure_seed)

        self.adjacency_matrix = nx.to_numpy_array(self.network)
        self.sparse_adjacency_matrix = sp.csr_matrix(self.adjacency_matrix)
        # Get the non-zero indices of the adjacency matrix
        self.row_indices_sparse, self.col_indices_sparse = self.sparse_adjacency_matrix.nonzero()

        #SCIPY ALT
        self.weighting_matrix = self._normlize_matrix(self.sparse_adjacency_matrix)
        #self.network_density = nx.density(self.network)

    ########################################################################################################

    def _generate_init_data_preferences_coherance(self) -> tuple[npt.NDArray, npt.NDArray]:
        np.random.seed(self.preferences_seed)#For inital construction set a seed
        preferences_beta = np.random.beta( self.a_preferences, self.b_preferences, size=self.N*self.M)# THIS WILL ALWAYS PRODUCE THE SAME OUTPUT
        preferences_capped_uncoherant = np.clip(preferences_beta, 0 + self.clipping_epsilon_init_preference, 1- self.clipping_epsilon_init_preference)

        np.random.seed(self.shuffle_coherance_seed)#For inital construction set a seed. THIS IS REALLY NTO VERY IMPORTANT
        if self.coherance_state != 0:
            #the issue seems to be here
            preferences_sorted = sorted(preferences_capped_uncoherant)#LIST IS NOW SORTED
            preferences_coherance = self._partial_shuffle_vector(np.asarray(preferences_sorted), self.shuffle_reps_coherance)
            low_carbon_preference_matrix = preferences_coherance.reshape(self.N,self.M)
        else:
            low_carbon_preference_matrix = preferences_capped_uncoherant.reshape(self.N,self.M)

        #LINE BELOW IS IMPORTANT!
        np.random.shuffle(low_carbon_preference_matrix)#THIS MIXES ROWS BUT NOT WITHIN THE ROW#THE IDEA IS TO HAVE ALL THE INDIVIDUALS MIXED BUT NOT THE ACTUAL PREFERENCES WITHIN EACH INDIVIDUAL

        return low_carbon_preference_matrix
    

    def _circular_list(self, list_to_circle) -> list:
        first_half = list_to_circle[::2]  # take every second element in the list, even indicies
        every_second_element = list_to_circle[1::2]# take every second element , odd indicies
        second_half = every_second_element[::-1] #reverse it
        circular_matrix = np.concatenate((first_half, second_half), axis=0)
        return circular_matrix
    
    def _partial_shuffle_matrix(self, matrix_to_shuffle, shuffle_reps) -> np.ndarray:
        for _ in range(shuffle_reps):
            a, b = np.random.randint(low=0, high=self.N, size=2)  # generate pair of indices to swap rows
            matrix_to_shuffle[[a, b], :] = matrix_to_shuffle[[b, a], :]
        return matrix_to_shuffle

    def _partial_shuffle_vector(self, vector_to_shuffle, shuffle_reps) -> np.ndarray:
        pairs = np.random.randint(low=0, high=len(vector_to_shuffle), size=(shuffle_reps,2) )
        for a,b in pairs:
            vector_to_shuffle[a], vector_to_shuffle[b] = vector_to_shuffle[b], vector_to_shuffle[a]
        return vector_to_shuffle
    
    def _shuffle_preferences_start_mixed(self): 
        low_carbon_preference_matrix_unsorted =  deepcopy(self.low_carbon_preference_matrix)
        identity_unsorted = np.mean(low_carbon_preference_matrix_unsorted, axis = 1)#TAKES MEAN WITHIN A ROW, IE THE IDENTITY 
        zipped_lists = zip(identity_unsorted, low_carbon_preference_matrix_unsorted)
        sorted_lists = sorted(zipped_lists, key=lambda x: x[0])#order by the identity 
        _ , sorted_preferences_tuple = zip(*sorted_lists)
        sorted_preferences = np.asarray(sorted_preferences_tuple)

        if (self.network_type== "SF") and (self.SF_green_or_brown_hegemony == 1):#WHY DOES IT ORDER IT THE WRONG WAY ROUND???
            sorted_preferences = sorted_preferences[::-1]
        elif (self.network_type== "SW"):
            sorted_preferences = self._circular_list(sorted_preferences)#agent list is now circular in terms of identity
        elif (self.network_type == "SBM"):
            pass

        np.random.seed(self.shuffle_homophily_seed)#Set seed for shuffle
        
        partial_shuffle_matrix = self._partial_shuffle_matrix(sorted_preferences, self.shuffle_reps_homophily)#partial shuffle of the list

        return partial_shuffle_matrix
    
    #################################################################################################
    
    def _update_carbon_price(self):
        if self.t == self.burn_in_duration:
            self.carbon_price_m = self.carbon_price_increased_m
            self.prices_high_carbon_instant = self.prices_high_carbon_m + self.carbon_price_m

    def _initialize_substitutabilities(self):
        self.low_carbon_substitutability = self.parameters["low_carbon_substitutability"]

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
        low_carbon_preferences = (1 - self.phi)*self.low_carbon_preference_matrix + self.phi*self.social_component_vector
        low_carbon_preferences_clipped  = np.clip(low_carbon_preferences, 0 + self.clipping_epsilon_init_preference, 1 - self.clipping_epsilon_init_preference)#this stops the guassian error from causing A to be too large or small thereby producing nans
        return low_carbon_preferences_clipped

    def _calc_Omega_m(self):
        omega_vector = ((self.prices_high_carbon_instant* self.low_carbon_preference_matrix) / (self.prices_low_carbon_m * (1 - self.low_carbon_preference_matrix))) ** self.low_carbon_substitutability
        return omega_vector

    def _calc_chi_m_nested_CES(self, Omega_m_matrix):
        chi_m = (((self.sector_preferences * self.low_carbon_preference_matrix)/(self.prices_low_carbon_m*Omega_m_matrix**(1/self.low_carbon_substitutability)))*(self.low_carbon_preference_matrix*Omega_m_matrix**((self.low_carbon_substitutability-1)/(self.low_carbon_substitutability)) + 1 - self.low_carbon_preference_matrix)**((self.sector_substitutability-self.low_carbon_substitutability)/(self.sector_substitutability*(self.low_carbon_substitutability-1))))** self.sector_substitutability
        return chi_m

    def _calc_Z(self,Omega_m_matrix, chi_m_tensor):
        common_vector = Omega_m_matrix * self.prices_low_carbon_m + self.prices_high_carbon_instant
        no_sum_Z_terms = chi_m_tensor * common_vector
        Z = no_sum_Z_terms.sum(axis=1)#sum with people
        return Z

    def _calc_consumption(self):
        Omega_m_matrix = self._calc_Omega_m()
        chi_m_tensor = self._calc_chi_m_nested_CES(Omega_m_matrix)
        Z_vec = self._calc_Z(Omega_m_matrix, chi_m_tensor)

        Z_matrix = np.tile(Z_vec, (self.M, 1)).T#repeat it so that can have chi tensor
        self.H_m_matrix = self.instant_expenditure * chi_m_tensor / Z_matrix
        self.L_m_matrix = Omega_m_matrix * self.H_m_matrix
        self.outward_social_influence_matrix = self._calc_consumption_ratio()

    
########################################################################################

    def _calc_ego_influence_degroot(self) -> npt.NDArray:
        # Perform the matrix multiplication using the dot method for sparse matrices
        neighbour_influence = self.weighting_matrix.dot(self.outward_social_influence_matrix)
        return neighbour_influence

    def _calc_consumption_ratio(self):
        ratio = self.L_m_matrix/(self.L_m_matrix + self.H_m_matrix)
        return ratio
    
    def _calc_social_component_matrix(self) -> npt.NDArray:
        if self.alpha_change_state in ("static_socially_determined_weights","dynamic_socially_determined_weights"):
            social_influence = self._calc_ego_influence_degroot_independent()
        else:#culturally determined either static or dynamic
            social_influence = self._calc_ego_influence_degroot()           
        return social_influence
    
    """
    def _calc_weighting_matrix_attribute(self, attribute_matrix):
        #Calculate the weighting matrix based on the Euclidean distance between agents' attributes.

        # Check if the attribute_matrix is a vector (1D array)
        if attribute_matrix.ndim == 1:
            # Compute the differences only for the non-zero entries
            summed_differences = np.abs(attribute_matrix[self.row_indices_sparse] - attribute_matrix[self.col_indices_sparse]) ** 2
        else:
            # For 2D matrix, compute the differences across each dimension
            differences = np.abs(attribute_matrix[self.row_indices_sparse, :] - attribute_matrix[self.col_indices_sparse, :]) ** 2
            # Sum the differences across all dimensions (axis=1 for 2D, no effect for 1D)
            summed_differences = np.sum(differences, axis=1)

        
        # Compute the Euclidean distance (or Minkowski distance with p=2)
        distances = summed_differences ** (1 / 2)
        
        # Compute the weighting values using the Euclidean distances
        weights = np.exp(-self.confirmation_bias * distances)
        
        # Create a sparse matrix with the same structure as the adjacency matrix
        non_diagonal_weighting_matrix = sp.csr_matrix(
            (weights, (self.row_indices_sparse, self.col_indices_sparse)),
            shape=self.adjacency_matrix.shape
        )
        
        # Normalize the matrix row-wise
        norm_weighting_matrix = self._normlize_matrix(non_diagonal_weighting_matrix)
        
        return norm_weighting_matrix
    """
    
    #"""
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
    #"""

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
        self.identity_vec = self._calc_identity(self.low_carbon_preference_matrix)#CAN REMOVE LATER, JUST TO KEEP TRACK OF IT
        return weighting_matrix_list  

    def _calc_emissions(self):
        self.total_carbon_emissions_flow = np.sum(self.H_m_matrix)# Calculate the emissions flow for the current time step
        self.total_carbon_emissions_stock += self.total_carbon_emissions_flow# Accumulate the emissions to compute the stock over time
    
    def _calc_carbon_dividend(self):
        total_quantities_m = np.sum(self.H_m_matrix,axis = 0)
        tax_income_R =  np.sum(self.carbon_price_m*total_quantities_m) 
        carbon_dividend = tax_income_R/self.N
        return carbon_dividend

    def _calc_instant_expediture(self):
        instant_expenditure = self.base_expenditure + self.carbon_dividend
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
    
    def _set_up_time_series(self):
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
        self.instant_expenditure = self._calc_instant_expediture()#update expenditures with previous steps gains

        if self.alpha_change_state != "fixed_preferences":
            self.low_carbon_preference_matrix = self._update_preferences()

        self._calc_consumption()

        if self.alpha_change_state != "fixed_preferences":
            if self.alpha_change_state == "dynamic_identity_determined_weights":
                self.weighting_matrix = self._update_weightings()
            elif self.alpha_change_state == "dynamic_socially_determined_weights":#independent behaviours
                self.weighting_matrix_tensor = self._update_weightings_list()
            else:
                pass #this is for "uniform_network_weighting", "static_socially_determined_weights","static_culturally_determined_weights"
            self.social_component_vector = self._calc_social_component_matrix()

        self.carbon_dividend = self._calc_carbon_dividend()
        
        if self.t > self.burn_in_duration:#what to do it on the end so that its ready for the next round with the tax already there
            self._calc_emissions()

        if self.save_timeseries_data_state:
            self.total_carbon_emissions_flow_vec = self.H_m_matrix.sum(axis = 1)
            if self.t == self.burn_in_duration + 1:#want to create it the step after burn in is finished
                self._set_up_time_series()
            elif (self.t % self.compression_factor_state == 0) and (self.t > self.burn_in_duration):
                self._save_timeseries_data_state_network()