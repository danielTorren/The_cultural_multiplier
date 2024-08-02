import numpy as np
import networkx as nx
from copy import deepcopy

import repast4py
from repast4py import core, schedule
from repast4py.space import Network
from repast4py.network import NetworkGenerator

# Define agent class
class NetworkAgent(core.Agent):
    def __init__(self, id, parameters):
        super().__init__(id)
        self.parameters = parameters
        self.initialize_state(parameters)

    def initialize_state(self, parameters):
        self.save_timeseries_data_state = parameters["save_timeseries_data_state"]
        self.compression_factor_state = parameters["compression_factor_state"]
        self.heterogenous_intrasector_preferences_state = parameters["heterogenous_intrasector_preferences_state"]
        self.heterogenous_carbon_price_state = parameters["heterogenous_carbon_price_state"]
        self.heterogenous_sector_substitutabilities_state = parameters["heterogenous_sector_substitutabilities_state"]
        self.heterogenous_phi_state = parameters["heterogenous_phi_state"]
        self.imitation_state = parameters["imitation_state"]
        self.alpha_change_state = parameters["alpha_change_state"]
        self.vary_seed_state = parameters["vary_seed_state"]
        self.network_type = parameters["network_type"]

        # seeds
        if self.vary_seed_state == "preferences":
            self.preferences_seed = int(round(parameters["set_seed"]))
            self.network_structure_seed = parameters["network_structure_seed"]
            self.shuffle_homophily_seed = parameters["shuffle_homophily_seed"]
            self.shuffle_coherance_seed = parameters["shuffle_coherance_seed"]
        elif self.vary_seed_state == "network":
            self.preferences_seed = parameters["preferences_seed"]
            self.network_structure_seed = int(round(parameters["set_seed"]))
            self.shuffle_homophily_seed = parameters["shuffle_homophily_seed"]
            self.shuffle_coherance_seed = parameters["shuffle_coherance_seed"]
        elif self.vary_seed_state == "shuffle_homophily":
            self.preferences_seed = parameters["preferences_seed"]
            self.network_structure_seed = parameters["network_structure_seed"]
            self.shuffle_homophily_seed = int(round(parameters["set_seed"]))
            self.shuffle_coherance_seed = parameters["shuffle_coherance_seed"]
        elif self.vary_seed_state == "shuffle_coherance":
            self.preferences_seed = parameters["preferences_seed"]
            self.network_structure_seed = parameters["network_structure_seed"]
            self.shuffle_homophily_seed = parameters["shuffle_homophily_seed"]
            self.shuffle_coherance_seed = int(round(parameters["set_seed"]))

        self.N = int(round(parameters["N"]))
        self.M = int(round(parameters["M"]))

        # Initialize network parameters
        self.initialize_network_parameters()

        # Initialize time
        self.t = 0
        self.burn_in_duration = parameters["burn_in_duration"]
        self.carbon_price_duration = parameters["carbon_price_duration"]
        self.time_step_max = self.burn_in_duration + self.carbon_price_duration

        # Initialize prices
        self.initialize_prices(parameters)

        # Initialize social learning and bias
        self.confirmation_bias = parameters["confirmation_bias"]
        self.clipping_epsilon_init_preference = parameters["clipping_epsilon_init_preference"]

        if self.heterogenous_phi_state:
            self.phi_array = np.linspace(parameters["phi_lower"], parameters["phi_upper"], num=self.M)
        else:
            self.phi_array = np.linspace(parameters["phi_lower"], parameters["phi_lower"], num=self.M)

        # Initialize network homophily
        self.initialize_homophily(parameters)

        self.individual_expenditure_array = np.asarray([1 / self.N] * self.N)
        self.instant_expenditure_vec = self.individual_expenditure_array
        self.sector_substitutability = parameters["sector_substitutability"]
        self.sector_preferences = np.asarray([1 / self.M] * self.M)

        if self.heterogenous_intrasector_preferences_state == 1:
            self.a_preferences = parameters["a_preferences"]
            self.b_preferences = parameters["b_preferences"]
            self.std_low_carbon_preference = parameters["std_low_carbon_preference"]
            self.low_carbon_preference_matrix_init = self.generate_init_data_preferences_coherance()
        elif self.heterogenous_intrasector_preferences_state == 0:
            self.low_carbon_preference_matrix_init = np.asarray([[1 / self.M] * self.M] * self.N)
        else:
            self.low_carbon_preference_matrix_init = np.asarray([np.random.uniform(size=self.M)] * self.N)

        self.low_carbon_preference_matrix = self.low_carbon_preference_matrix_init
        self.adjacency_matrix, self.weighting_matrix, self.network = self.create_weighting_matrix()

        self.update_carbon_price()
        self.identity_vec = self.calc_identity(self.low_carbon_preference_matrix)

        if self.homophily_state != 0:
            self.low_carbon_preference_matrix = self.shuffle_preferences_start_mixed()

        if (self.SBM_block_heterogenous_individuals_substitutabilities_state == 0) and (self.heterogenous_sector_substitutabilities_state == 1):
            self.low_carbon_substitutability_array = np.linspace(parameters["low_carbon_substitutability_lower"], parameters["low_carbon_substitutability_upper"], num=self.M)
            self.low_carbon_substitutability_matrix = np.asarray([self.low_carbon_substitutability_array] * self.N)
        elif (self.SBM_block_heterogenous_individuals_substitutabilities_state == 1) and (self.heterogenous_sector_substitutabilities_state == 0):
            block_substitutabilities = np.linspace(parameters["low_carbon_substitutability_lower"], parameters["low_carbon_substitutability_upper"], num=self.SBM_block_num)
            low_carbon_substitutability_matrix = np.tile(block_substitutabilities[:, np.newaxis], (1, self.M))
            self.low_carbon_substitutability_matrix = np.repeat(low_carbon_substitutability_matrix, self.SBM_block_sizes, axis=0)
        else:
            self.low_carbon_substitutability_array = np.linspace(parameters["low_carbon_substitutability_upper"], parameters["low_carbon_substitutability_upper"], num=self.M)
            self.low_carbon_substitutability_matrix = np.asarray([self.low_carbon_substitutability_array] * self.N)

        self.calc_consumption()

        if self.alpha_change_state == "fixed_preferences":
            self.social_component_matrix = self.low_carbon_preference_matrix
        else:
            if self.alpha_change_state in ("uniform_network_weighting", "static_culturally_determined_weights", "dynamic_identity_determined_weights", "common_knowledge_dynamic_identity_determined_weights"):
                self.weighting_matrix = self.update_weightings()
            elif self.alpha_change_state in ("static_socially_determined_weights", "dynamic_socially_determined_weights"):
                self.weighting_matrix_tensor = self.update_weightings_list()
            self.social_component_matrix = self.calc_social_component_matrix()

        self.carbon_dividend_array = self.calc_carbon_dividend_array()

        self.total_carbon_emissions_stock = 0
        self.total_carbon_emissions_stock_sectors = np.zeros(self.M)
        if self.network_type == "SBM":
            self.group_indices_list = self.calc_group_ids()
            self.total_carbon_emissions_stock_blocks = np.asarray([0] * self.SBM_block_num)

    def initialize_network_parameters(self):
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

    def initialize_prices(self, parameters):
        self.prices_low_carbon_m = np.asarray([1] * self.M)
        self.prices_high_carbon_m = np.asarray([1] * self.M)
        self.carbon_price_m = np.asarray([0] * self.M)

        if self.heterogenous_carbon_price_state:
            self.carbon_price_increased_m = np.linspace(parameters["carbon_price_increased_lower"], parameters["carbon_price_increased_upper"], num=self.M)
        else:
            self.carbon_price_increased_m = np.linspace(parameters["carbon_price_increased_lower"], parameters["carbon_price_increased_lower"], num=self.M)

    def initialize_homophily(self, parameters):
        self.homophily_state = parameters["homophily_state"]
        self.coherance_state = parameters["coherance_state"]
        self.coherance = parameters["coherance"]
        self.coherance_state_identity = parameters["coherance_state_identity"]
        self.coherance_identity = parameters["coherance_identity"]
        self.coherance_state_identity_choice = parameters["coherance_state_identity_choice"]
        self.coherance_identity_choice = parameters["coherance_identity_choice"]

    def generate_init_data_preferences_coherance(self):
        np.random.seed(self.preferences_seed)
        preferences = np.random.beta(self.a_preferences, self.b_preferences, (self.N, self.M))
        preferences /= np.sum(preferences, axis=1)[:, np.newaxis]
        return preferences

    def create_weighting_matrix(self):
        # Generate the network based on the specified type
        if self.network_type == "SW":
            graph = nx.watts_strogatz_graph(self.N, self.SW_K, self.SW_prob_rewire, seed=self.network_structure_seed)
        elif self.network_type == "SBM":
            sizes = [int(self.N / self.SBM_block_num)] * self.SBM_block_num
            intra_block = np.ones((self.SBM_block_num, self.SBM_block_num)) * self.SBM_network_density_input_intra_block
            np.fill_diagonal(intra_block, self.SBM_network_density_input_inter_block)
            graph = nx.stochastic_block_model(sizes, intra_block, seed=self.network_structure_seed)
        elif self.network_type == "BA":
            graph = nx.barabasi_albert_graph(self.N, self.BA_nodes, seed=self.network_structure_seed)
        else:
            raise ValueError("Unsupported network type")

        adjacency_matrix = nx.to_numpy_array(graph)
        weighting_matrix = adjacency_matrix / np.sum(adjacency_matrix, axis=1)[:, np.newaxis]
        return adjacency_matrix, weighting_matrix, graph

    def update_carbon_price(self):
        if self.t >= self.burn_in_duration:
            self.carbon_price_m = self.carbon_price_increased_m

    def calc_identity(self, preferences):
        return np.mean(preferences, axis=1)

    def shuffle_preferences_start_mixed(self):
        np.random.seed(self.shuffle_homophily_seed)
        shuffled_preferences = deepcopy(self.low_carbon_preference_matrix)
        np.random.shuffle(shuffled_preferences)
        return shuffled_preferences

    def update_weightings(self):
        if self.alpha_change_state == "uniform_network_weighting":
            return self.adjacency_matrix / np.sum(self.adjacency_matrix, axis=1)[:, np.newaxis]
        elif self.alpha_change_state == "static_culturally_determined_weights":
            return self.weighting_matrix
        elif self.alpha_change_state in ("dynamic_identity_determined_weights", "common_knowledge_dynamic_identity_determined_weights"):
            identity_diff = np.abs(self.identity_vec[:, np.newaxis] - self.identity_vec[np.newaxis, :])
            weighting_matrix = self.adjacency_matrix * (1 - identity_diff)
            weighting_matrix /= np.sum(weighting_matrix, axis=1)[:, np.newaxis]
            return weighting_matrix

    def update_weightings_list(self):
        weighting_matrix_tensor = []
        for t in range(self.time_step_max):
            weighting_matrix_tensor.append(self.update_weightings())
        return weighting_matrix_tensor

    def calc_social_component_matrix(self):
        return np.dot(self.weighting_matrix, self.low_carbon_preference_matrix)

    def calc_carbon_dividend_array(self):
        carbon_emissions = np.sum(self.instant_expenditure_vec * self.prices_high_carbon_m)
        carbon_tax_revenue = carbon_emissions * self.carbon_price_m.mean()
        return np.ones(self.N) * (carbon_tax_revenue / self.N)

    def calc_consumption(self):
        high_carbon_expenditure = self.instant_expenditure_vec * self.prices_high_carbon_m
        low_carbon_expenditure = self.instant_expenditure_vec * self.prices_low_carbon_m
        self.total_carbon_emissions_stock += np.sum(high_carbon_expenditure)
        self.total_carbon_emissions_stock_sectors += high_carbon_expenditure
        self.instant_expenditure_vec = low_carbon_expenditure / np.sum(low_carbon_expenditure)

    def calc_group_ids(self):
        block_ids = []
        for i, size in enumerate(self.SBM_block_sizes):
            block_ids.extend([i] * size)
        return block_ids

    def step(self):
        self.t += 1
        # Update steps for the agent
        self.update_carbon_price()
        self.identity_vec = self.calc_identity(self.low_carbon_preference_matrix)

        if self.homophily_state != 0:
            self.low_carbon_preference_matrix = self.shuffle_preferences_start_mixed()

        if self.alpha_change_state == "fixed_preferences":
            self.social_component_matrix = self.low_carbon_preference_matrix
        else:
            if self.alpha_change_state in ("uniform_network_weighting", "static_culturally_determined_weights", "dynamic_identity_determined_weights", "common_knowledge_dynamic_identity_determined_weights"):
                self.weighting_matrix = self.update_weightings()
            elif self.alpha_change_state in ("static_socially_determined_weights", "dynamic_socially_determined_weights"):
                self.weighting_matrix_tensor = self.update_weightings_list()
            self.social_component_matrix = self.calc_social_component_matrix()

        self.carbon_dividend_array = self.calc_carbon_dividend_array()
        self.calc_consumption()

############################################################################################################
############################################################################################################

# Main simulation function
def main(params):
    repast4py.initialize(sys.argv)

    context = core.Context("NetworkModel")

    # Create agents
    for i in range(params["num_agents"]):
        agent = NetworkAgent(i, params)
        context.add(agent)

    # Create networks
    network = Network("network", context)
    generator = NetworkGenerator()
    if params["network_type"] == "SW":
        generator.watts_strogatz(network, params["N"], params["SW_K"], params["SW_prob_rewire"], params["network_structure_seed"])
    elif params["network_type"] == "SBM":
        sizes = [params["N"] // params["SBM_block_num"]] * params["SBM_block_num"]
        intra_block = np.ones((params["SBM_block_num"], params["SBM_block_num"])) * params["SBM_network_density_input_intra_block"]
        np.fill_diagonal(intra_block, params["SBM_network_density_input_inter_block"])
        generator.stochastic_block_model(network, sizes, intra_block, params["network_structure_seed"])
    elif params["network_type"] == "BA":
        generator.barabasi_albert(network, params["N"], params["BA_nodes"], params["network_structure_seed"])

    # Scheduler
    scheduler = schedule.init_schedule_runner(context)

    def step():
        for agent in context.agents():
            agent.step()

    scheduler.schedule_repeating(step, interval=1)

    # Run the simulation
    scheduler.execute()

    repast4py.finalize()