## Folder structure:
Inside the package folder, you will find a set of folders that include the core model itself, running, plotting and other utility code. Additionally, there are two jupyter notebooks. The first of which, "produce_figures.ipynb", is a guide to reproduce the figures found in the paper; some of these require substantial run times. Secondly, there is "model_playground.ipynb" which allows you to test out a single model run for a variety of different parameter inputs and produce different plots to analyse that experiment.

## Outline of model:
The python files that the core model is built of may be found in package/model, network_matrix.py is the main manager of the simulation and holds an array of agent preferences, network structure and manages consumptio decisions. Each of the N individuals has preferences A for consumption in M categories; these preferences evolve due to repeated social imitation of consumption. The mean of these preferences is used to calculate an environmental identity. The distance between individuals' environmental identities determines how strong their connection is and, thus, how much attention is paid to that neighbour's consumption behaviours.

## Other folders in the package:
- "package/constants" contains several json files. "base_params.json" contains the default model parameters which are used to reproduce multiple figures. One dict files are used to study the variation of a single parameter (oneD_dict_networks_tax_sweep.json). Variable parameter json files which are used to set the ranges of parameter variations for the sensitivity analysis (variable_parameters_dict_SA.json) or which two parameters to vary to cover a 2D parameter space (twoD_dict_networks_tau_sub.json).

- "generating_data" contains several python files that load in inputs and run the model for said conditions, then save this data:
	- "example_networks_gen" produce examples of small-world, stochastic block model and scale free networks tested
 	- "example_timeseries_gen" produces examples of two experimental runs, one for no carbon price and another for low carbon price to give a feel for what the timeseries output of a typical run looks like
 	- "network_homo_gen" varies the degree of homophily and hegemony in the model to study the role of network structures in inhibiting decarbonisation. Over three different network structures. 
	- "network_multiplier_gen" calculates the tax reduction due to the cultural and social multipliers, relative to the fixed preferences counter case. Over three different network structures.
	- "network_tau_sub_gen" produces cumulative emissions varying both the carbon tax and low- and high-carbon substitutability for the case of the cultural multiplier. Over three different network structures. 
	- "network_tax_sweep_gen" produces cumulative emissions for different carbon tax values for three cases: fixed preferences, the social multiplier and the cultural multiplier. Over three different network structures. 
	- "network_tax_sweep_sub_gen" produces cumulative emissions for different carbon tax values and low- and three different high-carbon substitutability for three cases: fixed preferences, the social multiplier and the cultural multiplier. In the small world network
	- "sensitivity_analysis_gen.py" runs the model for a large number of parameter values and over multiple 
	- "single_experiment_gen.py" runs a single experiment	
	- "SW_M_gen" study how increasing the number of consumption categories affects cumulative emissions for the small-world network at three different carbon tax rates.

- "plotting_data" loads the model results created in the "generating_data" folder, analyses them and calls the plot functions.

- "resources" contains code that is used frequently such as saving or loading data (utility.py), running the simulation for a specific number of time steps (run.py)