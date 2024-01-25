import numpy as np
def calc_Omega_m(P_H,P_L,A_matrix, sigma_matrix):
    term_1 = (P_H*A_matrix)
    term_2 = (P_L*(1- A_matrix))
    omega_vector = (term_1/term_2)**(sigma_matrix)
    return omega_vector

def calc_n_tilde_m(A_matrix,Omega_m,sigma_matrix):
    n_tilde_m = (A_matrix*(Omega_m**((sigma_matrix-1)/sigma_matrix))+(1-A_matrix))**(sigma_matrix/(sigma_matrix-1))
    return n_tilde_m
    

def calc_chi_m_nested_CES(a,n_tilde_m,nu, P_H):
    chi_m = (a*(n_tilde_m**((nu-1)/nu)))/P_H
    return chi_m

def calc_Z(Omega_m,P_L,P_H,chi_m,nu):
    common_vector_denominator = Omega_m*P_L + P_H
    chi_pow = chi_m**nu
    Z = chi_pow*common_vector_denominator   
    return Z

def calculate_emissions(t_max, B, N, M, a, P_L, P_H, A_matrix, sigma_matrix, nu):
    a_matrix = np.tile(a, (N, 1))
    Omega_m = calc_Omega_m(P_H,P_L,A_matrix, sigma_matrix)
    n_tilde_m = calc_n_tilde_m(A_matrix,Omega_m,sigma_matrix)
    chi_m = calc_chi_m_nested_CES(a_matrix,n_tilde_m,nu, P_H)
    Z = calc_Z(Omega_m,P_L,P_H,chi_m,nu)
    H_m = (B*(chi_m**nu))/Z

    E = t_max*sum(H_m)

    return E