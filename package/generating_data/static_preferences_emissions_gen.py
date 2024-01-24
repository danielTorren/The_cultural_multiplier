import numpy as np

########################################################################################################################    
#CALCULATE THE PRICE EFFECT EMISSIONS BASED ON THE INITIAL CONDITIONS
    
def calculate_Z(chi, Omega, P_L, P_H, nu):
    return np.sum((chi ** nu) * (Omega * P_L + P_H), axis=1)

def calculate_chi(a, P_H, A, Omega, sigma, nu):
    return (a / P_H) * (A * Omega ** ((sigma - 1) / sigma) + (1 - A)) ** ((nu - 1) * sigma / (nu * (sigma - 1)))

def calculate_Omega(P_H, A, P_L, sigma):
    return ((P_H * A) / (P_L * (1 - A))) ** sigma

def calculate_emissions(t_max, B, N, M, a, P_L, P_H, A_matrix, sigma_matrix, nu):
    a_matrix = np.tile(a, (N, 1))
    #sigma_matrix = np.tile(sigma, (N, 1))

    Omega = calculate_Omega(P_H, A_matrix, P_L, sigma_matrix)
    chi = calculate_chi(a_matrix, P_H, A_matrix, Omega, sigma_matrix, nu)

    Z = calculate_Z(chi, Omega, P_L, P_H, nu)

    E = t_max * np.sum(1 / Z * np.sum(chi, axis=1))

    return E