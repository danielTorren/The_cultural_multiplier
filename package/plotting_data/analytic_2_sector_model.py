from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import check_other_folder

##############################################################################################################

def calculate_data(parameters, B, tau1):
    A1 = parameters["A1"]
    A2 = parameters["A2"]
    a1 = parameters["a1"]
    a2 = parameters["a2"]
    sigma1 = parameters["sigma1"]
    sigma2 = parameters["sigma2"]
    nu = parameters["nu"]
    PL1 = parameters["PL1"]
    PL2 = parameters["PL2"]
    PBH1 = parameters["PBH1"]
    PBH2 = parameters["PBH2"]
    h1 = parameters["h1"]
    h2 = parameters["h2"]
    tau2 = parameters["tau2"]
    
    Omega1 = ((PBH1 + tau1) * A1) / (PL1 * (1 - A1)) ** sigma1
    Omega2 = ((PBH2 + tau2) * A2) / (PL2 * (1 - A2)) ** sigma2
    
    chi1 = (a1 / (PBH1 + tau1)) * ((A1 * Omega1 ** ((sigma1 - 1) / sigma1)) + (1 - A1)) ** ((nu - 1) * sigma1 / (nu * (sigma1 - 1)))
    chi2 = (a2 / (PBH2 + tau2)) * ((A2 * Omega2 ** ((sigma2 - 1) / sigma2)) + (2 - A2)) ** ((nu - 1) * sigma2 / (nu * (sigma2 - 1)))
    
    Z = (chi1 ** nu) * (Omega1 * PL1 + PBH1 + tau1) + (chi2 ** nu) * (Omega2 * PL2 + PBH2 + tau2)
    
    H1 = ((chi1 ** nu) * (B - h1 * (PBH1 + tau1) - h2 * (PBH2 + tau2))) / Z + h1
    H2 = ((chi2 ** nu) * (B - h1 * (PBH1 + tau1) - h2 * (PBH2 + tau2))) / Z + h2
    
    E_F = H1 + H2

    derivative_E_F = (-1 / Z ** 2) * (B - h1 * (PBH1 + tau1) - h2 * (PBH2 + tau2)) * (chi1 ** nu + chi2 ** nu) +  (1 / Z) * (-h1 * (chi1 ** nu + chi2 ** nu) + (B - h1 * (PBH1 + tau1) - h2 * (PBH2 + tau2)) * 
                                   (nu * chi1 ** nu * ((nu - 1) / nu * (A1 * Omega1 ** ((sigma1 - 1) / sigma1)) / 
                                   (A1 * Omega1 ** ((sigma1 - 1) / sigma1) - (1 - A1)) - (1 - (sigma1 - 1) / sigma1)) / 
                                   ((1 - (sigma1 - 1) / sigma1) * (PBH1 + tau1))))
    
    return E_F, derivative_E_F


def gen_data_E(parameters, B_values, tau_values):
    B_values_grid, tau_values_grid = np.meshgrid(B_values, tau_values, indexing='ij')
    data_E_F, data_derv_E_F = calculate_data(parameters, B_values_grid, tau_values_grid)

    return data_E_F, data_derv_E_F

import matplotlib.pyplot as plt

def plot_contours(scen, B_values, tau_values, E_F_values, derivative_values, axis_val):
    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    fig.suptitle(scen)

    # Plot E_F
    ax1 = axs[0]
    if axis_val:
        CS1 = ax1.contourf(B_values, tau_values, E_F_values, cmap='viridis')
        #ax1.set_xlabel(r'Budget, $B$')
        ax1.set_ylabel(r'Sector 1 carbon price, $\tau_1$')
    else:
        CS1 = ax1.contourf(tau_values,B_values, E_F_values.T, cmap='viridis')
        #ax1.set_xlabel(r'Sector 1 carbon price, $\tau_1$')
        ax1.set_ylabel(r'Budget, $B$')
    cbar1 = fig.colorbar(CS1, ax=ax1)
    cbar1.set_label(r'Emissions Flow, $E_F$')

    #ax1.set_title('E_F')
    #ax1.set_xlabel(r'Budget, $B$')

    # Plot derivative of E_F with respect to tau1
    ax2 = axs[1]
    if axis_val:
        CS2 = ax2.contourf(B_values, tau_values, derivative_values, cmap='plasma')
        ax2.set_xlabel(r'Budget, $B$')
        ax2.set_ylabel(r'Sector 1 carbon price, $\tau_1$')
    else:
        CS2 = ax2.contourf(tau_values,B_values, derivative_values.T, cmap='plasma')
        ax2.set_xlabel(r'Sector 1 carbon price, $\tau_1$')
        ax2.set_ylabel(r'Budget, $B$')

    cbar2 = fig.colorbar(CS2, ax=ax2)
    cbar2.set_label(r'Derivative of Emissions Flow, $\frac{\partial E_F}{\partial \tau_1}$')

    #ax2.set_title('Derivative of E_F with respect to tau1')


    plt.tight_layout()

    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/2_sector_E_E_derv_" + scen + "_" + str(axis_val)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_contours_2_scen(B_values, tau_values, E_F_values1, derivative_values1, E_F_values2, derivative_values2, axis_val):
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey='row')

    # Plot E_F for Scenario 1
    ax1 = axs[0][0]
    if axis_val:
        CS1 = ax1.contourf(B_values, tau_values, E_F_values1, cmap='viridis')
        ax1.set_ylabel(r'Sector 1 carbon price, $\tau_1$')
    else:
        CS1 = ax1.contourf(tau_values, B_values, E_F_values1.T, cmap='viridis')
        ax1.set_ylabel(r'Budget, $B$')


    ax1.set_title(r'Scenario 1')

    # Plot derivative of E_F with respect to tau1 for Scenario 1
    ax2 = axs[1][0]
    if axis_val:
        CS2 = ax2.contourf(B_values, tau_values, derivative_values1, cmap='plasma')
        ax2.set_xlabel(r'Budget, $B$')
        ax2.set_ylabel(r'Sector 1 carbon price, $\tau_1$')
    else:
        CS2 = ax2.contourf(tau_values, B_values, derivative_values1.T, cmap='plasma')
        ax2.set_xlabel(r'Sector 1 carbon price, $\tau_1$')
        ax2.set_ylabel(r'Budget, $B$')

    # Plot E_F for Scenario 2
    ax3 = axs[0][1]
    if axis_val:
        CS3 = ax3.contourf(B_values, tau_values, E_F_values2, cmap='viridis')
    else:
        CS3 = ax3.contourf(tau_values, B_values, E_F_values2.T, cmap='viridis')
    ax3.set_title(r'Scenario 2')
    cbar3 = fig.colorbar(CS3, ax=[ax1, ax3])
    cbar3.set_label(r'Emissions Flow, $E_F$')

    # Plot derivative of E_F with respect to tau1 for Scenario 2
    ax4 = axs[1][1]
    if axis_val:
        CS4 = ax4.contourf(B_values, tau_values, derivative_values2, cmap='plasma')
        ax4.set_xlabel(r'Budget, $B$')
    else:
        CS4 = ax4.contourf(tau_values, B_values, derivative_values2.T, cmap='plasma')
        ax4.set_xlabel(r'Sector 2 carbon price, $\tau_1$')

    # Create a single color bar for each row
    cbar4 = fig.colorbar(CS4, ax=[ax2, ax4])
    cbar4.set_label(r'Derivative of Emissions Flow, $\frac{\partial E_F}{\partial \tau_1}$')

    #plt.tight_layout()

    # Save plot
    plotName = "results/Other"
    f = plotName + "/double_2_sector_E_E_derv_" + str(axis_val)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")



if __name__ == '__main__':
    """
    Scenario 1: is that sector 1 is a basic good with high subsititutability(FOOD), similar prices between both goods but a minimum of the high carbon required. But there is low preference for this sector
    Sector 2 has low substitutability, but more attractive (LONG DISTANCE TRAVEL), also the High carbon base price is much lower, howeer there is no minimum required
    - Assume individuals are indifferent to the environment
    """

    parameters_1 = {
        "A1": 0.5,
        "A2": 0.5,
        "tau2": 0,
        "a1": 0.5,
        "a2": 0.5,
        "sigma1": 5,
        "sigma2": 1.5,
        "nu": 1.5,
        "PL1": 1,
        "PL2": 1,
        "PBH1": 1,
        "PBH2": 1,
        "h1": 1,
        "h2": 0
    }
    """
    Scenario 2: is the opposite of Scenario 1 we now look at how taxing the second sector affects emissions.
    """
    parameters_2 = {
        "A1": parameters_1["A1"],#same
        "A2": parameters_1["A2"],#same
        "tau2": parameters_1["tau2"],#same
        "nu": parameters_1["nu"],#same
        "a1": parameters_1["a2"],
        "a2": parameters_1["a1"],
        "sigma1": parameters_1["sigma2"],
        "sigma2": parameters_1["sigma1"],
        "PL1": parameters_1["PL2"],
        "PL2": parameters_1["PL1"],
        "PBH1": parameters_1["PBH2"],
        "PBH2": parameters_1["PBH1"],
        "h1": parameters_1["h2"],
        "h2": parameters_1["h1"]
    }

    tau_values = np.linspace(0,1,1000)
    B_values = np.linspace(3,10,1000)
   

    #MAKE SURE THAT BUDGET IS LARGE ENOUGH TO SUPPORT THE MINIMUM QUANTITIES IN EACH SECTOR
    max_tau1 = max(tau_values)
    min_B_1 = parameters_1["h1"] * (parameters_1["PBH1"] + max_tau1) + parameters_1["h2"] * (parameters_1["PBH2"] + parameters_1["tau2"])
    min_B_2 = parameters_2["h1"] * (parameters_2["PBH1"] + max_tau1) + parameters_2["h2"] * (parameters_2["PBH2"] + parameters_2["tau2"])
    #print("min_B",  min_B)
    largest_min = max([min_B_1, min_B_2])
    print("largest_min B",largest_min)
    if (min(B_values) < largest_min):
        B_values = np.linspace(largest_min,max(B_values),len(B_values))
        print("Adjusted,min_B",  largest_min)
    
    data_E_F_1, data_derv_E_F_1 = gen_data_E(parameters_1, B_values, tau_values)
    data_E_F_2, data_derv_E_F_2 = gen_data_E(parameters_2, B_values, tau_values)
    
    axis_val = 0
    #plot_contours("Scenario 1",B_values, tau_values, data_E_F_1, data_derv_E_F_1, axis_val)
    #plot_contours("Scenario 2",B_values, tau_values, data_E_F_2, data_derv_E_F_2, axis_val)
    plot_contours_2_scen(B_values, tau_values, data_E_F_1, data_derv_E_F_1, data_E_F_2, data_derv_E_F_2, axis_val)

    plt.show()


