from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import createFolder,produce_name_datetime,save_object

def calculate_A_t(A_0, P_H, sigma, T):
    # Initialize a list to store the values of A_t for each time step
    A_values = [A_0]

    # Iterate from t = 1 to T
    for t in range(1, T + 1):
        A_t_minus_1 = A_values[-1]
        A_t_minus_2 = A_values[-2] if t >= 2 else A_0

        numerator = 1 + ((1 - (1 / (1 + ((1 - A_t_minus_2) / (P_H[t - 2] * A_t_minus_2))**sigma)))
                        / (P_H[t - 1] * (1 / (1 + ((1 - A_t_minus_2) / (P_H[t - 2] * A_t_minus_2))**sigma))))

        denominator = 1 + ((1 - (1 / (1 + ((1 - A_t_minus_1) / (P_H[t - 1] * A_t_minus_1))**sigma)))
                          / (P_H[t] * (1 / (1 + ((1 - A_t_minus_1) / (P_H[t - 1] * A_t_minus_1))**sigma))))

        A_t = A_t_minus_1 * (numerator / denominator)
        A_values.append(A_t)
        print("A_t",A_t)

    return A_values

def calc_A_init(A_j,P_H,sigma):
    A = (P_H**sigma)/((P_H**sigma)+(1/A_j - 1)**sigma)
    return A

def calculate_A_t_plus_1_alt(A_0, P_H, sigma, T):
    # Initialize a list to store the values of A_t_plus_1 for each time step
    A_values = [A_0,calc_A_init(A_0,P_H[0],sigma)]#

    print("A_values",A_values)
    # Iterate from t = 1 to T
    for t in range(1, T + 1):
        numerator = P_H[t]**sigma
        term_1 = (  (1/P_H[t-1])*((1/A_values[t-1]) -1) )**(sigma**2)
        denominator = P_H[t]**sigma + term_1
        A_t_plus_1 = A_values[t] * (numerator / denominator)
        A_values.append(A_t_plus_1)
        print("A_t",A_t_plus_1)

    return A_values

def calc_PH_values(P_init, P_final, num_timesteps):
    if num_timesteps < 2:
        raise ValueError("Number of time steps should be at least 2 for linear increase.")

    # Calculate the step size for the linear increase
    step_size = (P_final - P_init) / (num_timesteps - 1)

    # Create a vector of prices with a linear increase
    PH_values = [P_init + i * step_size for i in range(num_timesteps)]

    return PH_values

def calculate_A_values_alt_2(P_H, sigma, A_0_2,A_0_1, T):
    A_values = [None] * (T + 1)

    #initial values
    t = 0
    A_values[t] = A_0_1
    print(" A_values[t]", t, A_values[t])
    # Initialize A_{1,1} using the given initial condition
    t = 1
    A_values[t] = P_H[t-1]**sigma/(P_H[t-1]**sigma + ((1/A_0_2) - 1)**(sigma**2))
    print(" A_values[t]", t, A_values[t])
    # Iterate from t = 2 to T
    for t in range(2, T + 1):
        A_values[t] = P_H[t - 1]**sigma / (P_H[t - 1]**sigma + ((1 / A_values[t - 2]) - 1)**(sigma**2))
        print(" A_values[t]", t, A_values[t])

    return A_values

def calculate_A_values_alt_3(P_H, sigma, phi, A_0_1, A_0_2, T):
    A_1_values = [None] * (T + 1)
    A_2_values = [None] * (T + 1)

    # Initialize A_{1,1} and A_{1,2} using the given initial conditions
    A_1_values[0] = A_0_1
    A_2_values[0] = A_0_2

    t_series = [0]
    # Iterate from t = 1 to T
    for t in range(1, T + 1):
        try:
            A_1_values[t] = (1 - phi) * A_1_values[t - 1] + phi * (P_H[t - 1]**sigma / (P_H[t - 1]**sigma + (1 / A_2_values[t - 1] - 1)**sigma))
            A_2_values[t] = (1 - phi) * A_2_values[t - 1] + phi * (P_H[t - 1]**sigma / (P_H[t - 1]**sigma + (1 / A_1_values[t - 1] - 1)**sigma))
        except FloatingPointError:
            print("broke, last vals",A_1_values[-1], A_2_values[-1])
        t_series.append(t)
    return A_1_values, A_2_values, t_series

def moving_average(a, n=3):
    """unweighted movign average"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
def calc_H(T, B, P_H, A):
    H = [B/(((P_H[t]*A[t])/(1-A[t]))**sigma + P_H[t]) for t in range(T)]
    return H

if __name__ == '__main__':
    np.seterr(all="warn")
    # Example usage
    A_0_1 = 0.6  # Initial condition, used for A_1_2
    A_0_2 = 0.2  # Initial condition, used for A_1_1

    sigma = 10  # Example value for sigma (adjust as needed)
    tau = 2000  # Number of time steps
    P_init = 1.0
    carbon_tax_increase = 0.5

    P_final = P_init + carbon_tax_increase
    P_H_t = calc_PH_values(P_init, P_final, tau+1)
    #print("P_H_t", P_H_t)

    """
    print("START A1")
    result_1 = calculate_A_t_plus_1_alt(A_0_2, P_H, sigma, tau)
    print("START A2")
    result_2 = calculate_A_t_plus_1_alt(A_0_1, P_H, sigma, tau)
    """

    """
    print("START A1")
    result_1 = calculate_A_values_alt_2(P_H_t, sigma,A_0_2,A_0_1, tau)
    print("START A2")
    result_2 = calculate_A_values_alt_2(P_H_t, sigma, A_0_1,A_0_2,tau)
    """

    root = "test_simple_model"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)
    
    createFolder(fileName)

    phi_array = np.linspace(0.35, 0.6, 4) 

    fig, axes = plt.subplots(nrows=len(phi_array),ncols=2, constrained_layout = True)
    
    result_1_list = []
    result_2_list = []
    t_series_list = []

    for i,phi in enumerate(phi_array):
        result_1, result_2, t_series = calculate_A_values_alt_3(P_H_t, sigma, phi, A_0_1, A_0_2, tau)   
        result_1_list.append(result_1)
        result_2_list.append(result_2)
        t_series_list.append(t_series)
        

    for i,phi in enumerate(phi_array):
        axes[i][0].plot(t_series_list[i], result_1_list[i], label= "$\phi$ = %s" % (round(phi,3)), color = "Blue")
        axes[i][1].plot(t_series_list[i], result_2_list[i], label= "$\phi$ = %s" % (round(phi,3)), color = "Orange")
        axes[i][0].legend()
        axes[i][1].legend()

    axes[0][0].set_title("Agent 1")
    axes[0][1].set_title("Agent 2")

    plotName = fileName + "/Prints"
    f = plotName + "/full_data"
    fig.savefig(f + ".png", dpi=600, format="png")


    fig_2, axes_2 = plt.subplots(nrows=len(phi_array),ncols=2, constrained_layout = True)
    rolling_window = 10
    for i,phi in enumerate(phi_array):
        rolling_mean_1 = moving_average(result_1_list[i],rolling_window)
        rolling_mean_2 = moving_average(result_2_list[i],rolling_window)
        time_1 = range(rolling_window,len(rolling_mean_1) + rolling_window)
        time_2 = range(rolling_window,len(rolling_mean_2)+ rolling_window)
        #print("time",time_1)
        axes_2[i][0].plot(time_1, rolling_mean_1, label= "$\phi$ = %s" % (round(phi,3)), color = "Blue")
        axes_2[i][1].plot(time_2, rolling_mean_2, label= "$\phi$ = %s" % (round(phi,3)), color = "Orange")
        axes_2[i][0].legend()
        axes_2[i][1].legend()

    axes_2[0][0].set_title("Agent 1 ROLLING window =  %s" % rolling_window)
    axes_2[0][1].set_title("Agent 2 ROLLING window =  %s" % rolling_window)

    plotName = fileName + "/Prints"
    f = plotName + "/rolling_window_%s" %(rolling_window)
    fig_2.savefig(f + ".png", dpi=600, format="png")


    B_1 = 100
    B_2 = 100
    fig_3, axes_3 = plt.subplots(nrows=len(phi_array),ncols=2, constrained_layout = True)
    for i,phi in enumerate(phi_array):
        rolling_mean_1 = calc_H(len(t_series_list[i]), B_1, P_H_t, result_1_list[i])
        rolling_mean_2 = calc_H(len(t_series_list[i]), B_2, P_H_t, result_1_list[i])
        #print("time",time_1)
        axes_3[i][0].plot(t_series_list[i], rolling_mean_1, label= "$\phi$ = %s" % (round(phi,3)), color = "Blue")
        axes_3[i][1].plot(t_series_list[i], rolling_mean_2, label= "$\phi$ = %s" % (round(phi,3)), color = "Orange")
        axes_3[i][0].legend()
        axes_3[i][1].legend()

    axes_3[0][0].set_title("Agent 1 high carbon consumption")
    axes_3[0][1].set_title("Agent 2 high carbon consumption")

    plotName = fileName + "/Prints"
    f = plotName + "/high_carbon_consumption"
    fig_3.savefig(f + ".png", dpi=600, format="png")


    plt.show()
