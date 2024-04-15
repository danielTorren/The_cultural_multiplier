import numpy as np
import matplotlib.pyplot as plt

def plot_equation():
    # Define the range of values for tau_1
    tau_1 = np.linspace(0, 1, 1000)
    
    # Define the equation
    E_F = (2 + tau_1)**2 / (((2 + tau_1)**2 + 2) * (1 + tau_1)**2) + 1 / ((2 + tau_1)**2 + 2)
    E_F_1 = (2 + tau_1)**2 / (((2 + tau_1)**2 + 2) * (1 + tau_1)**2)
    E_F_2 = 1 / ((2 + tau_1)**2 + 2)
   


    # Plot the equation
    plt.plot(tau_1, E_F, label='Original Equation (Blue)')
    plt.plot(tau_1, E_F_1, 'orange', label='Additional Equation 1 (Orange)')
    plt.plot(tau_1, E_F_2, 'green', label='Additional Equation 2 (Green)')
    plt.xlabel(r'$\tau_1$')
    plt.ylabel(r'$E$')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_equation()

