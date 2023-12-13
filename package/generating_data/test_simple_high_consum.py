import numpy as np
import matplotlib.pyplot as plt

def calculate_H(A, sigma):
    """
    Calculate H using the given equation.

    Parameters:
    - A: Input value
    - sigma: Parameter value

    Returns:
    - H: Result of the equation
    """
    numerator = (1 - A)**sigma
    denominator = A**sigma + (1 - A)**sigma
    H = numerator / denominator
    return H

def calc_L(H, A,sigma):
    L = ((A/(1-A))**sigma)*H
    return L

def plot_H_against_A(sigma_values,A_values):
    """
    Plot H against A for different values of sigma.

    Parameters:
    - sigma_values: List of sigma values to be plotted

    Returns:
    - None (Displays the plot)
    """
   
    fig,axes = plt.subplots(ncols=4, figsize = (8,5), constrained_layout=True)

    for sigma in sigma_values:
        H_values = calculate_H(A_values, sigma)
        L_values = calc_L(H_values,A_values,sigma)
        L_values_expenditure = 1 - H_values
        axes[0].plot(A_values, H_values, label=f'$\sigma$={sigma}')
        axes[1].plot(A_values, L_values, label=f'$\sigma$={sigma}')
        axes[2].plot(H_values, L_values, label=f'$\sigma$={sigma}')
        axes[3].plot(H_values, L_values_expenditure, label=f'$\sigma$={sigma}')

    #fig.tight_layout()
    axes[0].set_title('H vs A for Different Sigma Values')
    axes[0].set_xlabel('A')
    axes[0].set_ylabel('H')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title('L vs A for Different Sigma Values')
    axes[1].set_xlabel('A')
    axes[1].set_ylabel('L')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_title('L vs H for Different Sigma Values')
    axes[2].set_xlabel('H')
    axes[2].set_ylabel('L')
    axes[2].legend()
    axes[2].grid(True)

    axes[3].set_title('L expenditure vs H for Different Sigma Values')
    axes[3].set_xlabel('H')
    axes[3].set_ylabel('L expenditure')
    axes[3].legend()
    axes[3].grid(True)

    f = "results/H_L_isoquant_vars"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def calculate_Omega(A, sigma):
    """
    Calculate Omega using the given equation.

    Parameters:
    - A: Input value
    - sigma: Parameter value

    Returns:
    - Omega: Result of the equation
    """
    Omega = (A / (1 - A))**sigma
    return Omega

def plot_Omega_against_A(sigma_values,A_values):
    """
    Plot Omega against A for different values of sigma.

    Parameters:
    - sigma_values: List of sigma values to be plotted

    Returns:
    - None (Displays the plot)
    """

    fig,ax = plt.subplots()

    for sigma in sigma_values:
        Omega_values = calculate_Omega(A_values, sigma)
        ax.plot(A_values, Omega_values, label=f'Sigma={sigma}')

    fig.tight_layout()
    ax.set_title('Omega vs A for Different Sigma Values')
    ax.set_xlabel('A')
    ax.set_ylabel('Omega')
    ax.legend()
    ax.grid(True)
    
    
    f = "results/omega_vars"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def calculate_log_Omega(A, sigma):
    """
    Calculate log(Omega) using the given equation.

    Parameters:
    - A: Input value
    - sigma: Parameter value

    Returns:
    - log_Omega: Result of the equation
    """
    log_Omega = sigma * (np.log(A) - np.log(1 - A))
    return log_Omega

def plot_log_Omega_against_log_A(sigma_values, A_values):
    """
    Plot log(Omega) against log(A) for different values of sigma.

    Parameters:
    - sigma_values: List of sigma values to be plotted

    Returns:
    - None (Displays the plot)
    """
    fig,ax = plt.subplots()

    for sigma in sigma_values:
        log_Omega_values = calculate_log_Omega(A_values, sigma)
        plt.plot(A_values, log_Omega_values, label=f'Sigma={sigma}')

    fig.tight_layout()
    ax.set_title('log(Omega) vs A for Different Sigma Values')
    ax.set_xlabel('A')
    ax.set_ylabel('log(Omega)')
    ax.legend()
    ax.grid(True)
    
    
    f = "results/log_omega_vars"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def calculate_log_Omega(A, sigma):
    """
    Calculate log(Omega) using the given equation.

    Parameters:
    - A: Input value
    - sigma: Parameter value

    Returns:
    - log_Omega: Result of the equation
    """
    log_Omega = sigma * (np.log(A) - np.log(1 - A))
    return log_Omega

def  calculate_L(sigma, A, U, H_vals):
    L = (               (1/A)*(     U**((sigma -1)/(sigma)) - (1-A)*H_vals**((sigma -1)/(sigma)))                      )**((sigma)/(sigma -1))
    return L 

def plot_H_L(sigma_values, A, U, H_vals):

    fig,ax = plt.subplots()

    for sigma in sigma_values:
        L_vals = calculate_L(sigma, A, U, H_vals)
        plt.plot(H_vals, L_vals, label=f'Sigma={sigma}')

    fig.tight_layout()
    ax.set_xlabel('H')
    ax.set_ylabel('L')
    ax.legend()
    ax.grid(True)
    
    f = "results/H_L_util"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

# Example: Plotting for sigma values 0.5, 1, 2, 3, and 4
sigma_values_to_plot = np.asarray([1.1,1.8,2,10,100])#np.linspace(0.5,3,7)
#A_values = np.linspace(0.1, 0.9, 100)  # Range of A values from 0 to 1
A = 0.5
U = 1
H_vals = np.linspace(1, 1000, 1000)
#plot_H_against_A(sigma_values_to_plot,A_values)
#plot_Omega_against_A(sigma_values_to_plot,A_values)
#plot_log_Omega_against_log_A(sigma_values_to_plot, A_values)
plot_H_L(sigma_values_to_plot, A, U, H_vals)

plt.show()

