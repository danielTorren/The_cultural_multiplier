import numpy as np
import networkx as nx
from copy import deepcopy
import scipy.sparse as sp

class Network_Matrix:
    def __init__(self, parameters: dict):
        """
        Initialize the Network_Matrix with given simulation parameters.

        Args:
            parameters (dict): Dictionary containing all necessary simulation parameters.

                Keys:
                - save_timeseries_data_state (int): If 1, time-series data will be saved; 0 if not.
                - compression_factor_state (int): Compression factor applied to stored data, reducing memory usage.
                - seed_reps (int): Number of seed repetitions for stochastic simulation consistency.
                - carbon_price_duration (int): Duration (in time steps) for which the carbon price remains active.
                - burn_in_duration (int): Initial duration for system to reach stability before main simulation starts. Set to zero
                - N (int): Number of agents in the network.
                - M (int): Number of sectors/categories represented in the network.
                - sector_substitutability (float): Elasticity of substitution between different sectors.
                - low_carbon_substitutability (float): Elasticity of substitution for low-carbon alternatives.
                - a_preferences (float): Beta function parameter used to create initial preference level for consumption.
                - b_preferences (float):  Beta function parameter used to create initial preference level for consumption.
                - clipping_epsilon_init_preference (float): Small value to prevent zero preferences, ensuring stability.
                - confirmation_bias (float): Degree of confirmation bias in agents' decision-making processes.
                - init_carbon_price (float): Initial carbon price at the start of the simulation, set to 0
                - increased_carbon_price (float): Carbon tax value
                - phi (float): Influence of social network on individual preferences, ranging from 0 to 1.
                - homophily_state (float): Degree of homophily in the network (0 = none, 1 = strong).
                - coherance_state (float): Degree of coherence within agents regarding  preferences.
                - SF_density (float): Density parameter for Scale-Free (SF) network structure.
                - SF_green_or_brown_hegemony (int): Initial dominance in network (0 = neutral, 1 = pro-environment, -1 = anti-environment).
                - SBM_block_num (int): Number of blocks or communities in the Stochastic Block Model (SBM) network.
                - SBM_network_density_input_intra_block (float): Intra-block density for SBM network.
                - SBM_network_density_input_inter_block (float): Inter-block density for SBM network.
                - SW_network_density (float): Density of Small-World (SW) network structure.
                - SW_prob_rewire (float): Probability of rewiring edges in the SW network to increase randomness.
        """
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
        self._initialize_intra_sector_preferences()
        self._update_carbon_price()
        self.identity_vec = self._calc_identity(self.low_carbon_preference_matrix)
        self._initialize_substitutabilities()
        self._calc_consumption()

        
        if self.alpha_change_state != "fixed_preferences":
            self._initialize_network_homophily()
            self._create_network()
            if self.homophily_state != 0:
                self.low_carbon_preference_matrix = self._shuffle_preferences_start_mixed()
    

        self._initialize_social_component()
        self.carbon_dividend = self._calc_carbon_dividend()
        self.total_carbon_emissions_stock = 0

    def _set_seeds(self):
        """Set random seeds for various components of the simulation."""
        self.preferences_seed = self.parameters["preferences_seed"]
        self.network_structure_seed = self.parameters["network_structure_seed"]
        self.shuffle_homophily_seed = self.parameters["shuffle_homophily_seed"]
        self.shuffle_coherance_seed = self.parameters["shuffle_coherance_seed"]

    def _set_state_attributes(self):
        """Initialize state attributes from parameters."""
        self.save_timeseries_data_state = self.parameters["save_timeseries_data_state"]
        self.compression_factor_state = self.parameters["compression_factor_state"]
        self.alpha_change_state = self.parameters["alpha_change_state"]

        if self.alpha_change_state not in ["dynamic_socially_determined_weights","fixed_preferences","dynamic_identity_determined_weights"]:
            raise ValueError(f"Invalid alpha change state")
    
        self.network_type = self.parameters["network_type"]
        
        self.N = int(round(self.parameters["N"]))
        self.M = int(round(self.parameters["M"]))

        self.shuffle_intensity = 1

    def _initialize_network_params(self):
        """
        Initialize network parameters based on the network type.
        
        Supports three types of networks:
        - SW: Small World network
        - SBM: Stochastic Block Model
        - SF: Scale Free network
        """
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

    def _calculate_nodes_SF(self) -> int:
        """
        Calculate the number of nodes for Scale Free network initialization.
        
        Returns:
            int: Number of nodes to use in Scale Free network generation
        """
        term1 = 2 * self.N - 1
        term2 = ((2 * self.N - 1) ** 2 - 4 * self.SF_density * self.N * (self.N - 1))**0.5
        nodes = (term1 - term2) / 2
        return int(round(nodes))

    def _initialize_time_params(self):
        """Initialize time-related parameters for the simulation."""
        self.t = 0
        self.burn_in_duration = self.parameters["burn_in_duration"]
        self.carbon_price_duration = self.parameters["carbon_price_duration"]
        self.time_step_max = self.burn_in_duration + self.carbon_price_duration

    def _initialize_prices(self):
        """Initialize price parameters for low-carbon and high-carbon products."""
        self.prices_low_carbon_m = 1
        self.prices_high_carbon_m = 1
        self.carbon_price_m = 0
        self.carbon_price_increased_m = self.parameters["carbon_price_increased"]

    def _initialize_social_learning(self):
        """Initialize parameters related to social learning and preference updates."""
        self.confirmation_bias = self.parameters["confirmation_bias"]
        self.clipping_epsilon_init_preference = self.parameters["clipping_epsilon_init_preference"]
        self.phi = self.parameters["phi"]

    def _initialize_preference_coherance(self):
        """
        Initialize preference coherence parameters.
        
        Coherence determines how similar preferences are within individual agents across sectors.
        """
        self.coherance_state = self.parameters["coherance_state"]
        self.shuffle_reps_coherance = int(round(((self.N*self.M) * (1 - self.coherance_state)) ** self.shuffle_intensity))

    def _initialize_network_homophily(self):
        """
        Initialize network homophily parameters.
        
        Homophily determines how likely agents are to connect with others who have similar envrionemtnal identity.
        """
        self.homophily_state = self.parameters["homophily_state"]
        self.shuffle_reps_homophily = int(round((self.N * (1 - self.homophily_state)) ** self.shuffle_intensity))

    def _initialize_expenditure(self):
        """Initialize agent expenditure parameters."""
        self.base_expenditure = 1/self.N
        self.instant_expenditure = self.base_expenditure

    def _initialize_sector_preferences(self):
        """Initialize sector-level preference parameters."""
        self.sector_substitutability = self.parameters["sector_substitutability"]
        self.sector_preferences = 1/ self.M

    def _initialize_intra_sector_preferences(self):
        """Initialize individual preferences within sectors."""
        self.a_preferences = self.parameters["a_preferences"]
        self.b_preferences = self.parameters["b_preferences"]
        self.low_carbon_preference_matrix_init = self._generate_init_data_preferences_coherance()
        self.low_carbon_preference_matrix = self.low_carbon_preference_matrix_init

    def _create_network(self):
        """
        Create the social network structure based on the specified network type.
        
        Supported network types:
        - SW: Small World network (Watts-Strogatz)
        - SBM: Stochastic Block Model
        - SF: Scale Free network (Barabási-Albert)
        """
        if self.network_type == "SW":
            self.network = nx.watts_strogatz_graph(n=self.N, k=self.SW_K, p=self.SW_prob_rewire, 
                                                 seed=self.network_structure_seed)
        elif self.network_type == "SBM":
            self.SBM_block_sizes = self._split_into_groups()
            block_probs = np.full((self.SBM_block_num, self.SBM_block_num), 
                                self.SBM_network_density_input_inter_block)
            np.fill_diagonal(block_probs, self.SBM_network_density_input_intra_block)
            self.network = nx.stochastic_block_model(sizes=self.SBM_block_sizes, p=block_probs, 
                                                   seed=self.network_structure_seed)
            self.block_id_list = np.asarray([i for i, size in enumerate(self.SBM_block_sizes) 
                                           for _ in range(size)])
        elif self.network_type == "SF":
            self.network = nx.barabasi_albert_graph(n=self.N, m=self.SF_nodes, 
                                                  seed=self.network_structure_seed)

        self.adjacency_matrix = nx.to_numpy_array(self.network)
        self.sparse_adjacency_matrix = sp.csr_matrix(self.adjacency_matrix)
        self.row_indices_sparse, self.col_indices_sparse = self.sparse_adjacency_matrix.nonzero()
        self.weighting_matrix = self._normlize_matrix(self.sparse_adjacency_matrix)


    def _generate_init_data_preferences_coherance(self) -> np.ndarray:
        """
        Generate initial preference data with specified coherence level.
        
        Returns:
            np.ndarray: Matrix of initial preferences with shape (N, M)
        """
        np.random.seed(self.preferences_seed)
        preferences_beta = np.random.beta(self.a_preferences, self.b_preferences, size=self.N*self.M)
        preferences_capped_uncoherant = np.clip(preferences_beta, 
                                              0 + self.clipping_epsilon_init_preference, 
                                              1 - self.clipping_epsilon_init_preference)

        np.random.seed(self.shuffle_coherance_seed)
        if self.coherance_state != 0:
            preferences_sorted = sorted(preferences_capped_uncoherant)
            preferences_coherance = self._partial_shuffle_vector(np.asarray(preferences_sorted), 
                                                              self.shuffle_reps_coherance)
            low_carbon_preference_matrix = preferences_coherance.reshape(self.N,self.M)
        else:
            low_carbon_preference_matrix = preferences_capped_uncoherant.reshape(self.N,self.M)

        np.random.shuffle(low_carbon_preference_matrix)
        return low_carbon_preference_matrix

    def _circular_list(self, list_to_circle: np.ndarray) -> np.ndarray:
        """
        Transform a list into a circular arrangement.
        
        Args:
            list_to_circle (np.ndarray): Input array to be circularly arranged
            
        Returns:
            np.ndarray: Circularly arranged array
        """
        first_half = list_to_circle[::2]
        every_second_element = list_to_circle[1::2]
        second_half = every_second_element[::-1]
        circular_matrix = np.concatenate((first_half, second_half), axis=0)
        return circular_matrix
    
    def _partial_shuffle_matrix(self, matrix_to_shuffle: np.ndarray, shuffle_reps: int) -> np.ndarray:
        """
        Partially shuffle a matrix by swapping random pairs of rows.
        
        Args:
            matrix_to_shuffle (np.ndarray): Matrix to be partially shuffled
            shuffle_reps (int): Number of row swaps to perform
            
        Returns:
            np.ndarray: Partially shuffled matrix
        """
        for _ in range(shuffle_reps):
            a, b = np.random.randint(low=0, high=self.N, size=2)
            matrix_to_shuffle[[a, b], :] = matrix_to_shuffle[[b, a], :]
        return matrix_to_shuffle

    def _partial_shuffle_vector(self, vector_to_shuffle: np.ndarray, shuffle_reps: int) -> np.ndarray:
        """
        Partially shuffle a vector by swapping random pairs of elements.
        
        Args:
            vector_to_shuffle (np.ndarray): Vector to be partially shuffled
            shuffle_reps (int): Number of element swaps to perform
            
        Returns:
            np.ndarray: Partially shuffled vector
        """
        pairs = np.random.randint(low=0, high=len(vector_to_shuffle), size=(shuffle_reps,2))
        for a, b in pairs:
            vector_to_shuffle[a], vector_to_shuffle[b] = vector_to_shuffle[b], vector_to_shuffle[a]
        return vector_to_shuffle
    
    def _shuffle_preferences_start_mixed(self) -> np.ndarray:
        """
        Create a mixed preference matrix based on network type and homophily settings.
        
        For SF networks, optionally reverses the order for green/brown hegemony.
        For SW networks, creates a circular arrangement.
        
        Returns:
            np.ndarray: Mixed preference matrix with shape (N, M)
        """
        low_carbon_preference_matrix_unsorted = deepcopy(self.low_carbon_preference_matrix)
        identity_unsorted = np.mean(low_carbon_preference_matrix_unsorted, axis=1)
        zipped_lists = zip(identity_unsorted, low_carbon_preference_matrix_unsorted)
        sorted_lists = sorted(zipped_lists, key=lambda x: x[0])
        _, sorted_preferences_tuple = zip(*sorted_lists)
        sorted_preferences = np.asarray(sorted_preferences_tuple)

        if (self.network_type == "SF") and (self.SF_green_or_brown_hegemony == 1):
            sorted_preferences = sorted_preferences[::-1]
        elif (self.network_type == "SW"):
            sorted_preferences = self._circular_list(sorted_preferences)

        np.random.seed(self.shuffle_homophily_seed)
        partial_shuffle_matrix = self._partial_shuffle_matrix(sorted_preferences, self.shuffle_reps_homophily)
        return partial_shuffle_matrix
    
    def _update_carbon_price(self):
        """
        Update carbon price if simulation has reached the burn-in duration.
        """
        if self.t == self.burn_in_duration:
            self.carbon_price_m = self.carbon_price_increased_m
            self.prices_high_carbon_instant = self.prices_high_carbon_m + self.carbon_price_m

    def _initialize_substitutabilities(self):
        """
        Initialize substitutability parameters for low-carbon options.
        """
        self.low_carbon_substitutability = self.parameters["low_carbon_substitutability"]

    def _initialize_social_component(self):
        """
        Initialize social component based on the alpha change state.
        
        Handles different weighting schemes:
        - Fixed preferences
        - Uniform network weighting
        - Static/dynamic culturally determined weights
        - Static/dynamic socially determined weights
        """
        if self.alpha_change_state == "fixed_preferences":
            self.social_component_vector = self.low_carbon_preference_matrix
        else:
            if self.alpha_change_state == "dynamic_identity_determined_weights":
                self.weighting_matrix = self._update_weightings()
            elif self.alpha_change_state == "dynamic_socially_determined_weights":
                self.weighting_matrix_tensor = self._update_weightings_list()

            self.social_component_vector = self._calc_social_component_matrix()

    def _update_preferences(self) -> np.ndarray:
        """
        Update agent preferences based on social influence and current preferences.
        
        Returns:
            np.ndarray: Updated preference matrix with values clipped to valid range
        """
        low_carbon_preferences = (1 - self.phi) * self.low_carbon_preference_matrix + self.phi * self.social_component_vector
        low_carbon_preferences_clipped = np.clip(low_carbon_preferences, 
                                               0 + self.clipping_epsilon_init_preference,
                                               1 - self.clipping_epsilon_init_preference)
        return low_carbon_preferences_clipped

    def _calc_Omega_m(self) -> np.ndarray:
        """
        Calculate the omega vector representing relative prices and preferences.
        
        Returns:
            np.ndarray: Omega matrix incorporating prices and preferences
        """
        omega_vector = ((self.prices_high_carbon_instant * self.low_carbon_preference_matrix) / 
                       (self.prices_low_carbon_m * (1 - self.low_carbon_preference_matrix))) ** self.low_carbon_substitutability
        return omega_vector

    def _calc_chi_m_nested_CES(self, Omega_m_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate chi values using nested CES (Constant Elasticity of Substitution) utility function.
        
        Args:
            Omega_m_matrix (np.ndarray): Matrix of omega values
            
        Returns:
            np.ndarray: Chi matrix for consumption calculations
        """
        chi_m = (((self.sector_preferences * self.low_carbon_preference_matrix) /
                 (self.prices_low_carbon_m * Omega_m_matrix**(1/self.low_carbon_substitutability))) *
                (self.low_carbon_preference_matrix * Omega_m_matrix**((self.low_carbon_substitutability-1)/
                                                                    (self.low_carbon_substitutability)) +
                 1 - self.low_carbon_preference_matrix)**((self.sector_substitutability-self.low_carbon_substitutability)/
                                                        (self.sector_substitutability*(self.low_carbon_substitutability-1)))
                ) ** self.sector_substitutability
        return chi_m

    def _calc_Z(self, Omega_m_matrix: np.ndarray, chi_m_tensor: np.ndarray) -> np.ndarray:
        """
        Calculate Z values for consumption normalization.
        
        Args:
            Omega_m_matrix (np.ndarray): Matrix of omega values
            chi_m_tensor (np.ndarray): Tensor of chi values
            
        Returns:
            np.ndarray: Vector of Z values
        """
        common_vector = Omega_m_matrix * self.prices_low_carbon_m + self.prices_high_carbon_instant
        no_sum_Z_terms = chi_m_tensor * common_vector
        Z = no_sum_Z_terms.sum(axis=1)
        return Z

    def _calc_consumption(self):
        """
        Calculate consumption patterns for all agents across sectors.
        Updates H_m_matrix (high-carbon consumption) and L_m_matrix (low-carbon consumption).
        """
        Omega_m_matrix = self._calc_Omega_m()
        chi_m_tensor = self._calc_chi_m_nested_CES(Omega_m_matrix)
        Z_vec = self._calc_Z(Omega_m_matrix, chi_m_tensor)

        Z_matrix = np.tile(Z_vec, (self.M, 1)).T
        self.H_m_matrix = self.instant_expenditure * chi_m_tensor / Z_matrix
        self.L_m_matrix = Omega_m_matrix * self.H_m_matrix
        self.outward_social_influence_matrix = self._calc_consumption_ratio()

    def _calc_consumption_ratio(self) -> np.ndarray:
        """
        Calculate ratio of low-carbon to total consumption.
        
        Returns:
            np.ndarray: Matrix of consumption ratios
        """
        ratio = self.L_m_matrix/(self.L_m_matrix + self.H_m_matrix)
        return ratio
    
    def _calc_social_component_matrix(self) -> np.ndarray:
        """
        Calculate social component based on network structure and influence type.
        
        Returns:
            np.ndarray: Matrix of social influences
        """
        if self.alpha_change_state == "dynamic_socially_determined_weights":
            social_influence = self._calc_ego_influence_degroot_independent()
        else:
            social_influence = self._calc_ego_influence_degroot()           
        return social_influence
    
    def _calc_weighting_matrix_attribute(self, attribute_array: np.ndarray) -> sp.csr_matrix:
        """
        Calculate weighting matrix based on attribute similarities.
        
        Args:
            attribute_array (np.ndarray): Array of attributes for calculating weights
            
        Returns:
            sp.csr_matrix: Sparse matrix of normalized weights
        """
        differences = attribute_array[self.row_indices_sparse] - attribute_array[self.col_indices_sparse]
        weights = np.exp(-self.confirmation_bias * np.abs(differences))
        non_diagonal_weighting_matrix = sp.csr_matrix(
            (weights, (self.row_indices_sparse, self.col_indices_sparse)),
            shape=self.adjacency_matrix.shape
        )
        norm_weighting_matrix = self._normlize_matrix(non_diagonal_weighting_matrix)
        return norm_weighting_matrix

    def _normlize_matrix(self, matrix: sp.csr_matrix) -> sp.csr_matrix:
        """
        Normalize a sparse matrix row-wise.
        
        Args:
            matrix (sp.csr_matrix): Sparse matrix to normalize
            
        Returns:
            sp.csr_matrix: Row-normalized sparse matrix
        """
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_row_sums = 1.0 / row_sums
        diagonal_matrix = sp.diags(inv_row_sums)
        norm_matrix = diagonal_matrix.dot(matrix)
        return norm_matrix

    def _calc_identity(self, low_carbon_preference_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate identity vector as mean of preferences across sectors.
        
        Args:
            low_carbon_preference_matrix (np.ndarray): Matrix of low-carbon preferences
            
        Returns:
            np.ndarray: Vector of identity values
        """
        return np.mean(low_carbon_preference_matrix, axis=1)
    
    def _calc_ego_influence_degroot(self) -> np.ndarray:
        """
        Calculate social influence using DeGroot learning model.
        
        Returns:
            np.ndarray: Matrix of social influences
        """
        neighbour_influence = self.weighting_matrix.dot(self.outward_social_influence_matrix)

        return neighbour_influence
    
    def _calc_ego_influence_degroot_independent(self) -> np.ndarray:
        """
        Calculate independent DeGroot learning influence for each sector.
        
        Returns:
            np.ndarray: Matrix of sector-specific social influences
        """
        neighbour_influence = np.zeros((self.N, self.M))
        if self.M > 1:
            for m in range(self.M):
                neighbour_influence[:, m] = self.weighting_matrix_tensor[m].dot(self.outward_social_influence_matrix[:, m])
        else:
            neighbour_influence = self.weighting_matrix_tensor.dot(self.outward_social_influence_matrix)#NOT ACTUALLY A TENSOR AS THERE IS ONLY ONE SECTOR

        return neighbour_influence
    
    def _update_weightings(self) -> sp.csr_matrix:
        """
        Update weighting matrix based on current identities.
        
        Returns:
            sp.csr_matrix: Updated weighting matrix
        """
        self.identity_vec = self._calc_identity(self.low_carbon_preference_matrix)
        norm_weighting_matrix = self._calc_weighting_matrix_attribute(self.identity_vec)
        return norm_weighting_matrix

    def _update_weightings_list(self) -> list:
        """
        Update weighting matrices for each sector independently.
        
        Returns:
            list: List of weighting matrices for each sector
        """
        #attribute_matrix = (self.outward_social_influence_matrix).T
        attribute_matrix = (self.low_carbon_preference_matrix).T
        if self.M == 1:
            norm_weighting_matrix = self._calc_weighting_matrix_attribute(attribute_matrix[0])
            return norm_weighting_matrix
        else:
            weighting_matrix_list = []
            for m in range(self.M):
                low_carbon_preferences_list = attribute_matrix[m]
                norm_weighting_matrix = self._calc_weighting_matrix_attribute(low_carbon_preferences_list)
                weighting_matrix_list.append(norm_weighting_matrix)
            self.identity_vec = self._calc_identity(self.low_carbon_preference_matrix)
            
            return weighting_matrix_list  

    def _calc_emissions(self):
        """
        Calculate and update carbon emissions flow and stock.
        """
        self.total_carbon_emissions_flow = np.sum(self.H_m_matrix)
        self.total_carbon_emissions_stock += self.total_carbon_emissions_flow
    
    def _calc_carbon_dividend(self) -> float:
        """
        Calculate carbon dividend distributed to agents.
        
        Returns:
            float: Per-agent carbon dividend
        """
        total_quantities_m = np.sum(self.H_m_matrix, axis=0)
        tax_income_R = np.sum(self.carbon_price_m * total_quantities_m) 
        carbon_dividend = tax_income_R/self.N
        return carbon_dividend

    def _calc_instant_expediture(self) -> float:
        """
        Calculate instant expenditure including carbon dividend.
        
        Returns:
            float: Updated expenditure value
        """
        instant_expenditure = self.base_expenditure + self.carbon_dividend
        return instant_expenditure

    def _split_into_groups(self) -> list:
        """
        Split agents into groups for SBM network creation.
        
        Returns:
            list: List of group sizes
        
        Raises:
            ValueError: If SBM_block_num is not positive
        """
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
        """
        Advance the simulation by one time step.
        
        Updates:
        - Time counter
        - Carbon prices
        - Agent expenditures
        - Preferences
        - Consumption patterns
        - Social learning weights
        - Carbon emissions
        - Historical data (if enabled)
        """
        self.t += 1
        self._update_carbon_price()
        self.instant_expenditure = self._calc_instant_expediture()

        if self.alpha_change_state != "fixed_preferences":
            self.low_carbon_preference_matrix = self._update_preferences()

        self._calc_consumption()

        if self.alpha_change_state != "fixed_preferences":
            if self.alpha_change_state == "dynamic_identity_determined_weights":
                self.weighting_matrix = self._update_weightings()
            elif self.alpha_change_state == "dynamic_socially_determined_weights":
                self.weighting_matrix_tensor = self._update_weightings_list()

            self.social_component_vector = self._calc_social_component_matrix()

        self.carbon_dividend = self._calc_carbon_dividend()
        
        if self.t > self.burn_in_duration:
            self._calc_emissions()

        if self.save_timeseries_data_state:
            self.total_carbon_emissions_flow_vec = self.H_m_matrix.sum(axis = 1)
            if self.t == self.burn_in_duration + 1:
                self._set_up_time_series()
            elif (self.t % self.compression_factor_state == 0) and (self.t > self.burn_in_duration):
                self._save_timeseries_data_state_network()