import numpy as np
import matplotlib.pyplot as plt
from seaborn import axes_style
from package.resources.utility import (
    load_object,
    save_object
)
from package.resources.run import generate_data
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed
import multiprocessing
from scipy.optimize import brentq
from package.generating_data.preference_complete_multiplier_tax_sweep_gen import process_seed, calc_M_vector_seeds

def main(
    fileName,
    tau_lower_bound = -0.9, 
    tau_upper_bound = 1000,
    total_range_runs = 100, 
    LOAD = False
) -> None:
    emissions_networks = load_object(fileName + "/Data","emissions_array")
    property_values_list = load_object(fileName + "/Data", "property_values_list")       
    base_params = load_object(fileName + "/Data", "base_params") 
    print("base_params", base_params["vary_seed_state"])

    if LOAD:
        tau_matrix = load_object(fileName + "/Data", "tau_matrix")
        emissions_matrix = load_object(fileName + "/Data", "emissions_matrix")
    else:
    
        seed_emissions_networks = np.transpose(emissions_networks, (3,0,1,2))

        results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(process_seed)(v, base_params, seed_emissions_networks, tau_lower_bound, tau_upper_bound, total_range_runs) for v in range(base_params["seed_reps"]))

        tau_data, emissions_data = zip(*results)
        tau_matrix = np.asarray(tau_data)
        emissions_matrix = np.asarray(emissions_data)

        print("CALCULATED DATA")
        save_object(tau_matrix,fileName + "/Data", "tau_matrix")
        save_object(emissions_matrix,fileName + "/Data", "emissions_matrix")


    M_vals_networks = calc_M_vector_seeds(property_values_list , tau_matrix, emissions_networks, emissions_matrix,base_params)
    save_object(M_vals_networks,fileName + "/Data", "M_vals_networks")

if __name__ == '__main__':
    plots = main(
        fileName= "results/phi_vary_09_14_29__17_05_2024",
        LOAD = False
    )