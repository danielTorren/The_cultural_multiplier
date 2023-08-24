from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import interp1d
from package.resources.utility import check_other_folder
from scipy.optimize import least_squares

#assume low carbon price is 1
def calc_H(A, P_H, tau, B, sigma):
    H = B/((P_H + tau) + ((P_H + tau )**sigma)*((A/(1-A))**sigma))
    return H 

def calc_L(H, A, P_H, tau,sigma):
    L = (((P_H+tau)*A/(1-A))**sigma)*H
    return L

def normlize_matrix(matrix):
    """
    Row normalize an array
    """
    row_sums = matrix.sum(axis=1)
    #print(matrix)
    #print("row_sums ",row_sums )
    norm_matrix = matrix / row_sums[:, np.newaxis]

    return norm_matrix 

def calc_weighting_matrix_attribute(attribute_array, adjacency_matrix,confirmation_bias):
    #test_arry = np.asarray([1,2,3,4,5])
    #print("test_arry",test_arry.shape)
    #print("attribute_array",attribute_array.shape)
    difference_matrix = np.subtract.outer(attribute_array, attribute_array) #euclidean_distances(attribute_array,attribute_array)# i think this actually not doing anything? just squared at the moment
    #print("difference_matrix",difference_matrix.shape)
    #quit()
    #print("confirmation_bias",confirmation_bias)
    #quit()
    alpha_numerator = np.exp(-np.multiply(confirmation_bias, np.abs(difference_matrix)))
    
    #print(" alpha_numerator", alpha_numerator)
    #quit()

    non_diagonal_weighting_matrix = (
        adjacency_matrix*alpha_numerator
    )  # We want onlythose values that have network connections
    #print("adjacency_matrix",adjacency_matrix.shape)
    #print("alpha_numerator",alpha_numerator.shape)
    #print("non_diagonal_weighting_matrix",non_diagonal_weighting_matrix.shape)
    #quit()

    norm_weighting_matrix = normlize_matrix(
        non_diagonal_weighting_matrix
    )  # normalize the matrix row wise

    return norm_weighting_matrix

def create_weighting_matrix(N, K, prob_rewire, set_seed):
    """
    Create watts-strogatz small world graph using Networkx library
    """

    G = nx.watts_strogatz_graph(n=N, k=K, p=prob_rewire, seed=set_seed)

    weighting_matrix = nx.to_numpy_array(G)

    norm_weighting_matrix = normlize_matrix(weighting_matrix)

    return (
        weighting_matrix,
        norm_weighting_matrix,
        G,
    )

def calc_social_components(A,H,L,mu,weighting_matrix):
    """
    Combine neighbour influence and social learning error to updated individual behavioural attitudes
    """           

    attribute_matrix = mu*A + (1 - mu)*(L/(L + H))
    #print("social influence attribute", attribute_matrix, attribute_matrix.shape)
    ##quit()
    #print("weighting_matrix",weighting_matrix,weighting_matrix.shape)
    social_influence = np.matmul(weighting_matrix, attribute_matrix)
    #print("social_influence",social_influence, social_influence.shape)
    #quit()
    return social_influence
    
def circular_agent_list(attribute_list) -> list:
    """
    Makes an ordered list circular so that the start and end values are matched in value and value distribution is symmetric

    """

    first_half = attribute_list[::2]  # take every second element in the list, even indicies
    second_half = (attribute_list[1::2])[::-1]  # take every second element , odd indicies
    attribute_list = first_half + second_half
    return attribute_list

def partial_shuffle_agent_list(attribute_list, shuffle_reps, N) -> list:
    """
    Partially shuffle a list using Fisher Yates shuffle
    """

    for _ in range(shuffle_reps):
        a, b = np.random.randint(
            low=0, high=N, size=2
        )  # generate pair of indicies to swap
        attribute_list[b], attribute_list[a] = attribute_list[a], attribute_list[b]
    
    return attribute_list

def shuffle_agent_list(attribute_list, shuffle_reps, N): 
    #make list cirucalr then partial shuffle it
    #print("before", attribute_list)
    attribute_list.sort()
    #print("after", attribute_list)
    attribute_list_circ = circular_agent_list(attribute_list)#agent list is now circular in terms of identity
    partial_attribute_list = partial_shuffle_agent_list(attribute_list_circ, shuffle_reps, N)#partial shuffle of the list
    
    return np.asarray(partial_attribute_list)


def alt_shuffle_agent_list(attribute_list, homophily, N): 
    #make list cirucalr then partial shuffle it
    #print("before", attribute_list)
    if homophily<0:
        #print("IN")
        np.random.shuffle(attribute_list)
    else:
        attribute_list.sort()
        #print("after", attribute_list)
        attribute_list_circ = circular_agent_list(attribute_list)#agent list is now circular in terms of identity
        shuffle_reps = int(round(N*(1 - homophily)))
        attribute_list = partial_shuffle_agent_list(attribute_list_circ, shuffle_reps, N)#partial shuffle of the list
    
    #print("partial_attribute_list",attribute_list)
    return np.asarray(attribute_list)

def calc_identity(low_carbon_preferences) -> float:
    identity = np.mean(low_carbon_preferences)
    return identity

def next_step(adjacency_matrix,A_init,H_init_array,L_init_array,mu,phi,tau,clipping_epsilon,confirmation_bias,P_H,B, sigma):
    I = [calc_identity(A_indiv) for A_indiv in A_init]
    #print("I",I)
    #quit()
    weighting_matrix_adjust = calc_weighting_matrix_attribute(I, adjacency_matrix,confirmation_bias)
    #print("weighting_matrix_adjust", weighting_matrix_adjust)
    #quit()
    #calc the social component for each individual
    S = calc_social_components(A_init,H_init_array,L_init_array,mu,weighting_matrix_adjust )
    #print("S",S)
    #update the preference
    A_new = (1-phi)*A_init + phi*S
    #print(" A_new ",  A_new )
    #calculate the consumption 
    H_new = calc_H(A_new, P_H, tau, B, sigma)# this is all the people consumption
    #print("post",H_new)
    H_new  = np.clip(H_new, 0 + clipping_epsilon, 1- clipping_epsilon)
    #print("post",H_new)
    L_new = calc_L(H_new,A_new, P_H, tau, sigma)
    
    return H_new,L_new,A_new



def sorted_calc_H_social(
    tau,
    phi,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    mu,
    homophily,
    identity_var,
    clipping_epsilon
):
    
    np.random.seed(set_seed)

    #calc shuffles
    

    #create attributes for agents
    I = np.random.beta(a_low_carbon_preference, b_low_carbon_preference, size=N)
    
    preferences_uncapped = np.asarray([np.random.normal(identity,identity_var, size=1)[0] for identity in  I])
    #print("preferences_uncapped",preferences_uncapped)
    A_init_unsorted = list(np.clip(preferences_uncapped, 0 + clipping_epsilon, 1 - clipping_epsilon))

    #print("A before shuffle",A)
    #shuffle_reps = int(round(N*(1 - homophily)))
    #A_init = shuffle_agent_list(A_init_unsorted, shuffle_reps, N)
    A_init = alt_shuffle_agent_list(A_init_unsorted, homophily, N)
    #print("A after shuffle",A_init)
    #print(A.shape)
    #quit()
    #sort the array

    H_init_array = calc_H(A_init, P_H, A_init, B, sigma)#in zeroth step tau = 0
    #print("H_init_array",H_init_array)
    L_init_array = calc_L(H_init_array,A_init, P_H, A_init, sigma)#in zeroth step tau = 0

    #create a network     
    adjacency_matrix, weighting_matrix, network= create_weighting_matrix(N, K, prob_rewire, set_seed)
    #print("adjacency_matrix",adjacency_matrix)
    #calc the weighting for each individual
    
    H_new, L_new,A_new = next_step(adjacency_matrix,A_init,H_init_array,L_init_array,mu,phi,tau,clipping_epsilon)

    #3 steps
    #H_new, L_new, A_new = next_step(adjacency_matrix,A_new,H_new, L_new,mu,phi,tau)
    #H_new, L_new, A_new = next_step(adjacency_matrix,A_new,H_new, L_new,mu,phi,tau)
    #H_new, L_new, A_new = next_step(adjacency_matrix,A_new,H_new, L_new,mu,phi,tau)
    
    #L_new = calc_L(H_init_array,A_new, P_H, tau, sigma)

    H_new_mean = np.mean(H_new)#is this what i want?
    #print("H mean",H_new_mean)

    return H_new_mean

def sort_homo_different_mu_tax_versus_quantities_for_social_preferences(
    mu_list,
    tau_list,
    phi_list,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    homophily,
    identity_var,
    clipping_epsilon
):

    H_list_taus = []

    for v in range(len(mu_list)):
        h_vec_mu = []
        for i in range(len(phi_list)):
            H_vec = []
            for j in range(len(tau_list)):
                H = sorted_calc_H_social(
                    tau_list[j],
                    phi_list[i],
                    N, 
                    K,
                    prob_rewire,
                    set_seed,
                    confirmation_bias,
                    a_low_carbon_preference,
                    b_low_carbon_preference,
                    P_H,  
                    B, 
                    sigma,
                    mu_list[v],
                    homophily,
                    identity_var,
                    clipping_epsilon
                    )
                H_vec.append(H)
            h_vec_mu.append(H_vec)
        H_list_taus.append(h_vec_mu)

    H_array_taus = np.asarray(H_list_taus)
    print(" H_array_taus", H_array_taus,H_array_taus.shape)
    
    fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True, figsize=(14, 7), sharex=True, sharey=True)

    for j, ax in enumerate(axes.flat):
        for i, phi in enumerate(phi_list):
            ax.plot(tau_list, H_array_taus[j][i], label = "$\phi$ = %s" %(round(phi,3)))
        ax.legend()
        ax.set_title("$\mu$ = %s" % (round(mu_list[j],3)))
        ax.set_xlabel(r"Carbon tax, $\tau$")#col
        #ax.xaxis.set_label_position('top') 
        ax.set_ylabel("High carbon quantity, H")#row

        #now plot the social multiplier
    fig.suptitle("P_H = %s, P_L = %s, B = %s, $\sigma$ = %s, N = %s, K = %s, $p_r$ = %s, a = %s, b = %s, $\\theta$ = %s, h = %s" % (P_H, P_L, B, sigma, N, K, prob_rewire , a_low_carbon_preference, b_low_carbon_preference, confirmation_bias,homophily))
    

    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/sorted_diif_mu_social_varying_tax_P_H_%s_sigma_%s" % (P_H, sigma) 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 
    

    # Create a new plot for the ratio
    fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True, figsize=(14, 7), sharex=True, sharey=True)

    for j, ax in enumerate(axes.flat):
        for i in range(len(phi_list) -1):
            fixed = H_array_taus[j][0]
            var = H_array_taus[j][i+1]
            y_values, x_reduction= trying_to_plot_theos_reduction(tau_list,var,fixed)
            ax.plot(y_values, x_reduction, label = "$\phi$ = %s" %(round(phi_list[i+1],3)))
        ax.legend()
        ax.set_title("$\mu$ = %s" % (round(mu_list[j],3)))
        ax.set_xlabel("High carbon quantity, H")#(r"Carbon tax, $\tau$")#col
        #ax.xaxis.set_label_position('top') 
        ax.set_ylabel("Tax reduction")

        #now plot the social multiplier
    fig.suptitle("P_H = %s, P_L = %s, B = %s, $\sigma$ = %s, N = %s, K = %s, $p_r$ = %s, a = %s, b = %s, $\\theta$ = %s, h = %s" % (P_H, P_L, B, sigma, N, K, prob_rewire , a_low_carbon_preference, b_low_carbon_preference, confirmation_bias,homophily))
    
    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/tax_reduct_sorted_diif_mu_social_varying_tax_P_H_%s_sigma_%s" % (P_H, sigma) 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

def trying_to_plot_theos_reduction(tau_list,y_line1,y_line2):

    # Create interpolation functions for both lines
    interp_line1 = interp1d(y_line1, tau_list, kind='linear')
    interp_line2 = interp1d(y_line2, tau_list, kind='linear')

    #print(tau_list,y_line1,y_line2)
    # Define the range of y values you want to consider
    y_min = max([min(y_line1)]+[min(y_line2)])
    y_max = min([max(y_line1)]+[max(y_line2)])

    #print("y_min", y_min)
    y_values = np.linspace(y_min, y_max, 100)

    # Calculate the x values for each y value using interpolation
    x_values_line1 = interp_line1(y_values)
    x_values_line2 = interp_line2(y_values)

    # Calculate the ratio of x values for each y value
    x_ratio = x_values_line1 / x_values_line2
    #print("x_ratio", x_ratio)
    x_reduction = 1 - x_ratio

    return y_values, x_reduction


def init_sorted_calc_H_social(
    N, 
    K,
    prob_rewire,
    set_seed,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    homophily,
    identity_var,
    clipping_epsilon
):
    
    np.random.seed(set_seed)

    #calc shuffles
    

    #create attributes for agents
    I = np.random.beta(a_low_carbon_preference, b_low_carbon_preference, size=N)
    
    preferences_uncapped = np.asarray([np.random.normal(identity,identity_var, size=1)[0] for identity in  I])
    #print("preferences_uncapped",preferences_uncapped)
    A_init_unsorted = list(np.clip(preferences_uncapped, 0 + clipping_epsilon, 1 - clipping_epsilon))

    #print("A before shuffle",A)
    #shuffle_reps = int(round(N*(1 - homophily)))
    #A_init = shuffle_agent_list(A_init_unsorted, shuffle_reps, N)
    A_init = alt_shuffle_agent_list(A_init_unsorted, homophily, N)
    #print("A after shuffle",A_init)
    #print(A.shape)
    #quit()
    #sort the array

    H_init_array = calc_H(A_init, P_H, A_init, B, sigma)#in zeroth step tau = 0
    #print("H_init_array",H_init_array)
    L_init_array = calc_L(H_init_array,A_init, P_H, A_init, sigma)#in zeroth step tau = 0

    #create a network     
    adjacency_matrix, weighting_matrix, network= create_weighting_matrix(N, K, prob_rewire, set_seed)
    #print("adjacency_matrix",adjacency_matrix)
    #calc the weighting for each individual
    
    return adjacency_matrix,A_init,H_init_array,L_init_array

def next_sorted_calc_H_social(adjacency_matrix,A_init,H_init_array,L_init_array,mu,phi,tau,clipping_epsilon,confirmation_bias,P_H,B, sigma):
    H_new, L_new,A_new = next_step(adjacency_matrix,A_init,H_init_array,L_init_array,mu,phi,tau,clipping_epsilon,confirmation_bias,P_H,B, sigma)
    return A_new, H_new, L_new

def calc_root(x, phi_val_root, Q_init, adjacency_matrix,A_new,H_new,L_new,mu,clipping_epsilon,confirmation_bias,P_H,B, sigma):
    A_new, H_new, L_new = next_sorted_calc_H_social(adjacency_matrix,A_new,H_new,L_new,mu,phi_val_root,x,clipping_epsilon,confirmation_bias,P_H,B, sigma)
    Q_run = sum(H_new)

    #if phi_val_root < 0.05:
    #    print("phi", phi_val_root)
    #    print(Q_run, Q_init/2,Q_run - Q_init/2)
    return Q_run - Q_init/2

def calc_M_given_phi_halving_emission(
    vars_dict,
    tau_guess_dyanmic,
    tau_guess_fixed
):
    
    xtol = 1e-6  # Example tolerance value
    #print("dyanmic")
    #print(vars_dict)
    
    adjacency_matrix = vars_dict["adjacency_matrix"]
    A_init = vars_dict["A_init"]
    H_init_array = vars_dict["H_init_array"]
    L_init_array = vars_dict["L_init_array"]
    mu = vars_dict["mu"]
    phi = vars_dict["phi"]
    #print("PHIII", phi)
    clipping_epsilon = vars_dict["clipping_epsilon"]
    confirmation_bias = vars_dict["confirmation_bias"]
    P_H = vars_dict["P_H"]
    B = vars_dict["B"]
    sigma = vars_dict["sigma"]
    A_init = params["A_init"]
    H_init_array = params["H_init_array"] 
    L_init_array = params["L_init_array"] 
    adjacency_matrix = params["adjacency_matrix"] 
    Q_init = params["Q_init"]
    
    result = least_squares(calc_root,verbose = 0, x0=tau_guess_dyanmic,xtol=xtol, bounds = (0, np.inf), args=(phi, Q_init, adjacency_matrix,A_init,H_init_array,L_init_array,mu,clipping_epsilon,confirmation_bias,P_H,B, sigma))
    tau_dynamic = result["x"][0]

    #print("now fixed")
    #NOW DO THE SAME BUT FOR FIXED PREFERENCES
    result_fixed = least_squares(calc_root, verbose = 0,x0=tau_guess_fixed,xtol=xtol, bounds = (0, np.inf), args=(0, Q_init, adjacency_matrix,A_init,H_init_array,L_init_array,mu,clipping_epsilon,confirmation_bias,P_H,B, sigma))
    tau_fixed = result_fixed["x"][0]

    #if phi > 0.05:
    #    quit()

    M = 1 - tau_dynamic/tau_fixed

    return M, tau_dynamic, tau_fixed

def phi_versus_M_gen_data(params, phi_list):

    adjacency_matrix,A_init,H_init_array,L_init_array= init_sorted_calc_H_social(
        params["N"], 
        params["K"],
        params["prob_rewire"],
        params["set_seed"],
        params["a_identity"],
        params["b_identity"],
        params["P_H"],  
        params["B"], 
        params["sigma"],
        params["homophily"],
        params["identity_var"],
        params["clipping_epsilon"]
        )
    Q_init = sum(H_init_array)

    data = []
    data_taus = []

    tau_dynamic_guess = 0 
    tau_fixed_guess  = 0

    params["A_init"] = A_init
    params["H_init_array"] = H_init_array
    params["L_init_array"] = L_init_array
    params["adjacency_matrix"] = adjacency_matrix
    params["Q_init"] = Q_init
    
    #print(params)
    #quit()
    counter = 0
    for phi_val in phi_list:
        params["phi"] = phi_val       
        M, tau_dynamic_guess, tau_fixed_guess = calc_M_given_phi_halving_emission(params,tau_dynamic_guess,tau_fixed_guess)
        data.append(M)
        data_taus.append((tau_dynamic_guess,tau_fixed_guess))
        counter +=1
        print(round((counter/len(phi_list))*100,3))
    return data, data_taus

def plot_phi_versus_M(data_100, data_1000,data_10000, phi_list):

    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(phi_list,data_100, label = "100")
    ax.scatter(phi_list,data_1000, label= "1000")
    ax.scatter(phi_list,data_10000, label= "10000")
    ax.legend()
    ax.set_xlabel(r"Phi")#col
    #ax.xaxis.set_label_position('top') 
    ax.set_ylabel("Tax reduction, M")#row

    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/tax_redux"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 


if __name__ == '__main__':
    """
    sigma = 3
    P_L = 1
    P_H = 1
    N = 100
    B = 1/N
    network_density = 0.3
    K = int(round((N - 1)*network_density))
    #print("K",K)
    #K = 4
    prob_rewire = 0.1
    set_seed = 1
    confirmation_bias = 0
    #a_low_carbon_preference = 2
    #b_low_carbon_preference = 2
    a_identity = 1
    b_identity = 1
    identity_var = 0.03
    #mu = 0
    homophily = -1
    clipping_epsilon = 1e-4
    """
    

    params = {
        "sigma": 2,
        "P_L": 1,
        "P_H": 1,
        "N": 1000,
        "B": 1,
        "network_density": 0.3,
        "prob_rewire": 0.1,
        "set_seed": 1,
        "confirmation_bias": 0,
        "a_identity": 1,
        "b_identity": 1,
        "identity_var": 0.03,
        "homophily": -1,
        "clipping_epsilon": 1e-4,
        "mu": 0,
        "confirmation_bias": 0
    }

    phi_list_6 = np.linspace(0,1, 100)

    #N = 100
    params["N"] =  100
    K = int(round(( params["N"] - 1)* params["network_density"]))
    params["K"] = K
    
    data_100, data_taus_100= phi_versus_M_gen_data(params, phi_list_6)
    
    params["N"] =  1000
    K = int(round(( params["N"] - 1)* params["network_density"]))
    params["K"] = K
    data_1000, data_taus_1000= phi_versus_M_gen_data(params, phi_list_6)

    params["N"] =  10000
    K = int(round(( params["N"] - 1)* params["network_density"]))
    params["K"] = K
    data_10000, data_taus_10000= phi_versus_M_gen_data(params, phi_list_6)
    #print("data_tau_dynamic, data_tau_fixed", data_taus)
    #print(params, phi_list_6)
    plot_phi_versus_M(data_100, data_1000,data_10000, phi_list_6)

    """
    tau_list_5 = np.linspace(0,5, 100)
    phi_list_5 = [0,0.33,0.66,1.0]#np.linspace(0, 1, 4)
    mu_list_5 = np.linspace(0, 1, 4)

    sort_homo_different_mu_tax_versus_quantities_for_social_preferences(
        mu_list_5,
        tau_list_5,
        phi_list_5,
        N, 
        K,
        prob_rewire,
        set_seed,
        confirmation_bias,
        #a_low_carbon_preference,
        #b_low_carbon_preference,
        a_identity,
        b_identity,
        P_H,  
        B, 
        sigma,
        homophily,
        identity_var,
        clipping_epsilon
    )

    #remake the phi plots
    """

    plt.show()






"""
def calc_H_social(
    tau,
    phi,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    mu
):
    
    np.random.seed(set_seed)

    #create attributes for agents
    A = np.random.beta(a_low_carbon_preference, b_low_carbon_preference, size=N)

    
    H_init_array = calc_H(A, P_H, tau, B, sigma)
    L_init_array = calc_L(H_init_array,A, P_H, tau, sigma)

    #create a network     
    adjacency_matrix, weighting_matrix, network= create_weighting_matrix(N, K, prob_rewire, set_seed)
    
    #calc the weighting for each individual

    
    weighting_matrix_adjust = calc_weighting_matrix_attribute(A, adjacency_matrix,confirmation_bias)

    #calc the social component for each individual
    S = calc_social_components(A,H_init_array,L_init_array,mu,weighting_matrix_adjust )

    #update the preference
    A_new = (1-phi)*A+ phi*S
    #calculate the consumption 
    H_new = calc_H(A_new, P_H, tau, B, sigma)# this is all the people consumption

    #L_new = calc_L(H_init_array,A_new, P_H, tau, sigma)

    H_new_mean = np.mean(H_new)#is this what i want?

    return H_new_mean

    
tau_list_1 = np.linspace(-0.5,0.5, 5)
preferences_list_1 = np.linspace(0.01,0.99, 100)
create_preferences_versus_quantities_for_different_tax(
    tau_list_1,
    preferences_list_1,
    P_H, 
    sigma,
    B,
    P_L
    )

tau_list_2 = np.linspace(-0.5,0.5, 100)
preferences_list_2 = np.linspace(0.01,0.99, 5)
create_tax_versus_quantities_for_preferences(
    tau_list_2,
    preferences_list_2,
    P_H, 
    sigma,
    B,
    P_L
    )


tau_list_3 = np.linspace(0,1, 100)
phi_list_3 = np.linspace(0, 1, 3)
create_tax_versus_quantities_for_social_preferences(
    tau_list_3,
    phi_list_3,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    mu
)


tau_list_4 = np.linspace(0,1, 100)
phi_list_4 = np.linspace(0, 1, 3)
mu_list_4 = np.linspace(0, 1, 4)
different_mu_tax_versus_quantities_for_social_preferences(
    mu_list_4,
    tau_list_4,
    phi_list_4,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma
)

mu = 0.0
tau = 0.5
phi = 0.5
bin_size= 0.01
histo_single(
    bin_size,
    mu,
    tau,
    phi,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    homophily
)


mu = 0.0
tau = 0.5
phi_list = np.asarray([0.3,0.6,0.9])
bin_size= 0.01
histo_multi(
    bin_size,
    mu,
    tau,
    phi_list,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    homophily
)
"""

"""

def create_preferences_versus_quantities_for_different_tax(
    tau_list,
    preferences_list
):

    H_list_taus = []
    L_list_taus = []
    Q_list_taus = []
    for tau in tau_list:
        H_vec = calc_H(preferences_list, P_H, tau, B, sigma)
        L_vec = calc_L(H_vec,preferences_list, P_H, tau, sigma)
        H_list_taus.append(H_vec)
        L_list_taus.append(L_vec)
        Q_list_taus.append(H_vec+L_vec)

    fig, axes = plt.subplots(ncols=3, nrows=1,constrained_layout=True, figsize=(14, 7))
    for i, tau in enumerate(tau_list):
        axes[0].plot(preferences_list, H_list_taus[i], label = "Tau = %s" %(round(tau,3)))
    #axes[0].legend()
    #axes[0].set_title("Fixed preferences")
    axes[0].set_xlabel("Low carbon preference, A")#col
    #ax.xaxis.set_label_position('top') 
    axes[0].set_ylabel("High carbon quantity, H")#row


    for i, tau in enumerate(tau_list):
        axes[1].plot(preferences_list, L_list_taus[i], label = "Tau = %s" %(round(tau,3)))
    #axes[1].legend()
    axes[1].set_title("P_H = %s,P_L = %s, B = %s, $\sigma$ = %s" % (P_H, P_L, B, sigma))
    axes[1].set_xlabel("Low carbon preference, A")#col
    #ax.xaxis.set_label_position('top') 
    axes[1].set_ylabel("Low carbon quantity, L")#row

    for i, tau in enumerate(tau_list):
        axes[2].plot(preferences_list, Q_list_taus[i], label = "Tau = %s" %(round(tau,3)))
    axes[2].legend()
    axes[2].set_xlabel("Low carbon preference, A")#col
    #ax.xaxis.set_label_position('top') 
    axes[2].set_ylabel("Consumption quantity, Q")#row

    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/fixed_preferences_tax_P_H_%s_sigma_%s" % (P_H, sigma) 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")  

def create_tax_versus_quantities_for_preferences(
    tau_list,
    preferences_list
):

    H_list_taus = []
    #L_list_taus = []
    #Q_list_taus = []
    for A in preferences_list:
        H_vec = calc_H(A, P_H, tau_list, B, sigma)
        #L_vec = calc_L(H_vec,preferences_list, P_H, tau, sigma)
        H_list_taus.append(H_vec)
        #L_list_taus.append(L_vec)
        #Q_list_taus.append(H_vec+L_vec)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(14, 7))
    for i, A in enumerate(preferences_list):
        ax.plot(tau_list, H_list_taus[i], label = "A = %s" %(round(A,3)))
    ax.legend()
    ax.set_title("P_H = %s,P_L = %s, B = %s, $\sigma$ = %s" % (P_H, P_L, B, sigma))
    ax.set_xlabel(r"Carbon tax, $\tau$")#col
    #ax.xaxis.set_label_position('top') 
    ax.set_ylabel("High carbon quantity, H")#row


    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/varying_tax_P_H_%s_sigma_%s" % (P_H, sigma) 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

def histo_single(
    bin_size,
    mu,
    tau,
    phi,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    homophily
    ):

    A_0 = all_sorted_calc_A_social(
        0,
        phi,
        N, 
        K,
        prob_rewire,
        set_seed,
        confirmation_bias,
        a_low_carbon_preference,
        b_low_carbon_preference,
        P_H,  
        B, 
        sigma,
        mu,
        homophily
    )
    A_tau = all_sorted_calc_A_social(
        tau,
        phi,
        N, 
        K,
        prob_rewire,
        set_seed,
        confirmation_bias,
        a_low_carbon_preference,
        b_low_carbon_preference,
        P_H,  
        B, 
        sigma,
        mu,
        homophily
    )

    # Calculate the bin edges
    min_val = np.min(0)
    max_val = np.max(1)
    num_bins = int((max_val - min_val) / bin_size) + 1
    bin_edges = np.linspace(min_val, max_val, num_bins)

    # Compute the histogram counts
    hist_counts_0, _ = np.histogram(A_0, bins=bin_edges)
    hist_counts_tau, _ = np.histogram(A_tau, bins=bin_edges)

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Plot the line histogram
    ax.plot(bin_edges[:-1], hist_counts_0, marker='o', linestyle='-', color='b', label = "before tax")
    ax.plot(bin_edges[:-1], hist_counts_tau, marker='o', linestyle='-', color='r', label = "after tax 0.5")
    ax.legend()
    # Add bars to the plot to represent the counts
    #ax.bar(bin_edges[:-1], hist_counts_0, width=bin_size, alpha=0.5, color='b')
    #ax.bar(bin_edges[:-1], hist_counts_tau, width=bin_size, alpha=0.5, color='b')


    # Add labels and title
    ax.set_xlabel('Preference,A')
    ax.set_ylabel('Frequency')
    #ax.set_title('Line Histogram with Bins of Size 0.05')

def all_sorted_calc_A_social(
    tau,
    phi,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    mu,
    homophily
):
    
    np.random.seed(set_seed)

    #calc shuffles
    shuffle_reps = int(round(N*(1 - homophily)))

    #create attributes for agents
    A = np.random.beta(a_low_carbon_preference, b_low_carbon_preference, size=N)
    A = shuffle_agent_list(list(A), shuffle_reps, N)
    #sort the array

    H_init_array = calc_H(A, P_H, tau, B, sigma)
    L_init_array = calc_L(H_init_array,A, P_H, tau, sigma)

    #create a network     
    adjacency_matrix, weighting_matrix, network= create_weighting_matrix(N, K, prob_rewire, set_seed)
    
    #calc the weighting for each individual

    
    weighting_matrix_adjust = calc_weighting_matrix_attribute(A, adjacency_matrix,confirmation_bias)

    #calc the social component for each individual
    S = calc_social_components(A,H_init_array,L_init_array,mu,weighting_matrix_adjust )

    #update the preference
    A_new = (1-phi)*A+ phi*S
    #calculate the consumption 
    #H_new = calc_H(A_new, P_H, tau, B, sigma)# this is all the people consumption

    #L_new = calc_L(H_init_array,A_new, P_H, tau, sigma)

    #H_new_mean = np.mean(H_new)#is this what i want?

    return A_new

def histo_multi(
    bin_size,
    mu,
    tau,
    phi_list,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    homophily
    ):

    # Set up the figure and axes
    fig, axes = plt.subplots(ncols= len(phi_list),constrained_layout=True, figsize=(14, 7), sharex=True)

    for i, ax in enumerate(axes.flat):
        A_0 = all_sorted_calc_A_social(
            0,
            phi_list[i],
            N, 
            K,
            prob_rewire,
            set_seed,
            confirmation_bias,
            a_low_carbon_preference,
            b_low_carbon_preference,
            P_H,  
            B, 
            sigma,
            mu,
            homophily
        )
        A_tau = all_sorted_calc_A_social(
            tau,
            phi_list[i],
            N, 
            K,
            prob_rewire,
            set_seed,
            confirmation_bias,
            a_low_carbon_preference,
            b_low_carbon_preference,
            P_H,  
            B, 
            sigma,
            mu,
            homophily
        )

        # Calculate the bin edges
        min_val = np.min(0)
        max_val = np.max(1)
        num_bins = int((max_val - min_val) / bin_size) + 1
        bin_edges = np.linspace(min_val, max_val, num_bins)

        # Compute the histogram counts
        hist_counts_0, _ = np.histogram(A_0, bins=bin_edges)
        hist_counts_tau, _ = np.histogram(A_tau, bins=bin_edges)

        # Plot the line histogram
        ax.plot(bin_edges[:-1], hist_counts_0, marker='o', linestyle='-', color='b', label = "before tax")
        ax.plot(bin_edges[:-1], hist_counts_tau, marker='o', linestyle='-', color='y', label = "after tax 0.5")
        ax.legend()
        # Add bars to the plot to represent the counts
        #ax.bar(bin_edges[:-1], hist_counts_0, width=bin_size, alpha=0.5, color='b')
        #ax.bar(bin_edges[:-1], hist_counts_tau, width=bin_size, alpha=0.5, color='b')


        # Add labels and title
        ax.set_xlabel('Preference,A')
        ax.set_ylabel('Frequency')
        ax.set_title('$\phi$ = %s' % (phi_list[i]))

def create_tax_versus_quantities_for_social_preferences(
    tau_list,
    phi_list,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma,
    mu
):

    #print("tau_list",tau_list)
    #print("phi_list",phi_list)

    H_list_taus = []

    for i in range(len(phi_list)):
        H_vec = []
        for j in range(len(tau_list)):
            H = calc_H_social(
                tau_list[j],
                phi_list[i],
                N, 
                K,
                prob_rewire,
                set_seed,
                confirmation_bias,
                a_low_carbon_preference,
                b_low_carbon_preference,
                P_H,  
                B, 
                sigma,
                mu
                )
            H_vec.append(H)
        H_list_taus.append(H_vec)

    H_array_taus = np.asarray(H_list_taus)
    #print(" H_array_taus", H_array_taus.shape)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(14, 7))

    for i, phi in enumerate(phi_list):
        ax.plot(tau_list, H_array_taus[i], label = "$\phi$ = %s" %(round(phi,3)))
    ax.legend()
    ax.set_title("P_H = %s, P_L = %s, B = %s, $\sigma$ = %s, N = %s, K = %s, $p_r$ = %s, $\mu$ = %s, a = %s, b = %s, $\\theta$ = %s" % (P_H, P_L, B, sigma, N, K, prob_rewire , mu, a_low_carbon_preference, b_low_carbon_preference, confirmation_bias))
    ax.set_xlabel(r"Carbon tax, $\tau$")#col
    #ax.xaxis.set_label_position('top') 
    ax.set_ylabel("High carbon quantity, H")#row

    #now plot the social multiplier


    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/social_varying_tax_P_H_%s_sigma_%s" % (P_H, sigma) 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

def different_mu_tax_versus_quantities_for_social_preferences(
    mu_list,
    tau_list,
    phi_list,
    N, 
    K,
    prob_rewire,
    set_seed,
    confirmation_bias,
    a_low_carbon_preference,
    b_low_carbon_preference,
    P_H,  
    B, 
    sigma
):

    H_list_taus = []

    for v in range(len(mu_list)):
        h_vec_mu = []
        for i in range(len(phi_list)):
            H_vec = []
            for j in range(len(tau_list)):
                H = calc_H_social(
                    tau_list[j],
                    phi_list[i],
                    N, 
                    K,
                    prob_rewire,
                    set_seed,
                    confirmation_bias,
                    a_low_carbon_preference,
                    b_low_carbon_preference,
                    P_H,  
                    B, 
                    sigma,
                    mu_list[v]
                    )
                H_vec.append(H)
            h_vec_mu.append(H_vec)
        H_list_taus.append(h_vec_mu)

    H_array_taus = np.asarray(H_list_taus)
    #print(" H_array_taus", H_array_taus.shape)

    fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True, figsize=(14, 7), sharex=True, sharey=True)

    for j, ax in enumerate(axes.flat):
        for i, phi in enumerate(phi_list):
            ax.plot(tau_list, H_array_taus[j][i], label = "$\phi$ = %s" %(round(phi,3)))
        ax.legend()
        ax.set_title("$\mu$ = %s" % (round(mu_list[j],3)))
        ax.set_xlabel(r"Carbon tax, $\tau$")#col
        #ax.xaxis.set_label_position('top') 
        ax.set_ylabel("High carbon quantity, H")#row

        #now plot the social multiplier
    fig.suptitle("P_H = %s, P_L = %s, B = %s, $\sigma$ = %s, N = %s, K = %s, $p_r$ = %s, a = %s, b = %s, $\\theta$ = %s" % (P_H, P_L, B, sigma, N, K, prob_rewire , a_low_carbon_preference, b_low_carbon_preference, confirmation_bias))
    

    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/diif_mu_social_varying_tax_P_H_%s_sigma_%s" % (P_H, sigma) 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 
"""