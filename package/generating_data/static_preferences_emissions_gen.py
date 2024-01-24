import numpy as np

########################################################################################################################    
#CALCULATE THE PRICE EFFECT EMISSIONS BASED ON THE INITIAL CONDITIONS
    
def calculate_Z(chi, Omega, P_L, P_H, nu):
    return np.sum((chi ** nu) * (Omega * P_L + P_H), axis=1)

def calculate_chi(a, P_H, A, Omega, sigma, nu):
    return (a / P_H) * (A * Omega ** ((sigma - 1) / sigma) + (1 - A)) ** ((nu - 1) * sigma / (nu * (sigma - 1)))

def calculate_Omega(P_H, A, P_L, sigma):
    return ((P_H * A) / (P_L * (1 - A))) ** sigma

def calculate_emissions(t_max, B, N, M, a, P_L, P_H, A, sigma, nu):
    a_matrix = np.tile(a, (N, 1))
    P_L_matrix = np.tile(P_L, (N, 1))
    P_H_matrix = np.tile(P_H, (N, 1))
    A_matrix = np.tile(A, (1, M))
    sigma_matrix = np.tile(sigma, (N, 1))
    nu_matrix = np.full_like(sigma_matrix, nu)

    Omega = calculate_Omega(P_H_matrix, A_matrix, P_L_matrix, sigma_matrix)
    chi = calculate_chi(a_matrix, P_H_matrix, A_matrix, Omega, sigma_matrix, nu_matrix)

    Z = calculate_Z(chi, Omega, P_L_matrix, P_H_matrix, nu_matrix)

    E = t_max * np.sum(1 / Z * np.sum(chi, axis=1))

    return E