import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from package.resources.plot import multiline
from matplotlib.cm import get_cmap
from joblib import Parallel, delayed
import multiprocessing

def calculate_Omega_m(P_Hm, A_m, P_Lm, sigma_m):
    Omega_m = ((P_Hm * A_m)/(P_Lm * (1 - A_m)))**sigma_m
    return Omega_m

def calculate_nm_tilde(A_m, Omega_m, psi_m):
    result = (A_m*(Omega_m**psi_m) + (1 - A_m))**(1 / psi_m)
    return result

def calculate_cm(Omega_m, P_Lm, P_Hm):
    cm = Omega_m * P_Lm + P_Hm
    return cm

def calculate_Z(a1, P_H1, n1_tilde, a2, P_H2, n2_tilde, omega):
    Z = ((a2*P_H1*n2_tilde)/(a1*P_H2*n1_tilde))**(1/(omega - 1))
    return Z

def calculate_root_H1(H1, omega, B, c1, c2, n1_tilde, n2_tilde, u1, u2,a2,P_H1,a1,P_H2):
    exponent = 1/(omega - 1)
    
    H2 = ((B - H1*c1)/ c2)
    print("H_2", H2, exponent,H1)
    term1 = (H1**exponent)*(H1*n1_tilde - u1)
    term2 = ((a2*P_H1*n2_tilde)/(a1*P_H2*n1_tilde))**(1/(omega - 1))
    term3 = H2**exponent
    term4 = H2*n2_tilde - u2
    result = term1 - term2*term3 * term4
    return result

def calc_psi(sigma_m):
    return (sigma_m -1)/sigma_m

def calc_omega(nu):
    return (nu -1)/nu

def calc_root(
    params
):
    a_1 = params["a_1"]
    a_2 = params["a_2"]#greater preference for flying over food
    P_H1 =params["P_H1"]#food more expensive?
    P_H2 =params["P_H2"]#flights cheap
    P_L1 = params["P_L1"]#unitary price
    P_L2 = params["P_L2"]#unitary price
    A_1 = params["A_1"]#-high green preference of low carbon food/high carbon
    A_2 = params["A_2"]#weak preferecen for low carbon airtransport? 
    sigma_1 = params["sigma_1"]#highish substitutability of low carbon food/energy
    sigma_2 = params["sigma_2"]#low substitutability of flying
    nu = params["nu"]
    B = params["B"]
    u_1 = params["u_1"]#need of food
    u_2 = params["u_2"]#no need for flying
    H1_0 = params["H1_0"]
    tau = params["tau"]#carbon tax
    
    print("INITAL CONDITIONS;", B, tau)
    if tau>0:
        P_H1 = P_H1 + tau
        P_H2 = P_H2 + tau

    psi_1 = calc_psi(sigma_1)
    Omega_1 = calculate_Omega_m(P_H1, A_1, P_L1, sigma_1)
    Omega_2 = calculate_Omega_m(P_H2, A_2, P_L2, sigma_2)
    c_1 = calculate_cm(Omega_1, P_L1, P_H1)
    #sanity_check 
    min_H_1 = u_1/((A_1*(Omega_1**psi_1) +(1-A_1))**(1/psi_1)) #use this as the initial guess?
    max_H_1 = B/c_1#what is the maximum amount of H_1 you could buy with your expenditure if you didnt get any H_2


    min_cost = P_H1*min_H_1 + P_L1*min_H_1*Omega_1#the min amount of H1 and therefore the min amount of L1
    
    #THIS IF STATEMENT DOESNT SEEM TO BE THE PROBLEM
    if B < min_cost:#if the expenditure isnt sufficent then do the best you can
        H_1 = min_H_1
        H_2 = 0
        print("insufficient dosh", B, min_cost)
    else:
        omega = calc_omega(nu)
        psi_2 = calc_psi(sigma_2)
        n1_tilde = calculate_nm_tilde(A_1, Omega_1, psi_1)#marginal psuedo utility with respect to H_1
        n2_tilde = calculate_nm_tilde(A_2, Omega_2, psi_2)#marginal psuedo utility with respect to H_2

        #print("intelligbel stuff = ",omega,psi_1,psi_2, Omega_1, Omega_2,n1_tilde, n2_tilde,)

        
        c_2 = calculate_cm(Omega_2, P_L2, P_H2)

        #print("opaque stuff", c_1, c_2, Z)

        print("inital guess low", min_H_1,calculate_root_H1(min_H_1,omega, B, c_1, c_2, n1_tilde, n2_tilde,u_1, u_2,a_2,P_H1,a_1,P_H2))
        print("initial guess high", max_H_1,calculate_root_H1(max_H_1,omega, B, c_1, c_2, n1_tilde, n2_tilde,u_1, u_2,a_2,P_H1,a_1,P_H2))

        solution = optimize.root(fun = calculate_root_H1, x0 = min_H_1, args=(omega, B, c_1, c_2, n1_tilde, n2_tilde,u_1, u_2,a_2,P_H1,a_1,P_H2))
        #print("solution", solution,solution["x"] ,solution,solution["x"][0])

        H_1 = solution["x"][0]
        H_2 = (B - H_1*c_1)/c_2

    emissions = H_1 + H_2
    L_1 = H_1*Omega_1
    L_2 = H_2*Omega_2
    low_carbon_consumption_prop = (L_1 + L_2)/(L_1 + L_2 + H_1 + H_2)

    #print("res", emissions, low_carbon_consumption_prop,tau,B)
    return emissions, low_carbon_consumption_prop

def multi_emissions_low_carbon(
        params_dict: list[dict]
):
    num_cores = multiprocessing.cpu_count()
    res = [calc_root(i) for i in params_dict]
    #res = Parallel(n_jobs=num_cores, verbose=10)(
    #    delayed(calc_root)(i) for i in params_dict
    #)
    emissions, low_carbon_consumption_prop = zip(
        *res
    )

    return np.asarray(emissions), np.asarray(low_carbon_consumption_prop)

def gen_params_dict(params_dict,expenditure_list, carbon_price_list, property_row, property_col):
    params_list = []
    for i in expenditure_list:
        for j in carbon_price_list:
            params_dict[property_row] = i
            params_dict[property_col] = j
            params_list.append(params_dict.copy())
    return params_list

def multi_line_matrix_plot(
    Z, col_vals, row_vals, cmap, col_axis_x, col_label, row_label, y_label
    ):

    fig, ax = plt.subplots( constrained_layout=True)#figsize=(14, 7)

    if col_axis_x:
        xs = np.tile(col_vals, (len(row_vals), 1))
        ys = Z
        c = row_vals

    else:
        xs = np.tile(row_vals, (len(col_vals), 1))
        ys = np.transpose(Z)
        c = col_vals
    
    #print("after",xs.shape, ys.shape, c.shape)

    ax.set_ylabel(y_label)#(r"First behaviour attitude variance, $\sigma^2$")
    
    #plt.xticks(x_ticks_pos, x_ticks_label)

    lc = multiline(xs, ys, c, cmap=cmap, lw=2)
    axcb = fig.colorbar(lc)

    if col_axis_x:
        axcb.set_label(row_label)#(r"Number of behaviours per agent, M")
        ax.set_xlabel(col_label)#(r'Confirmation bias, $\theta$')
        #ax.set_xticks(col_ticks_pos)
        #ax.set_xticklabels(col_ticks_label)
        #print("x ticks", col_label,col_ticks_pos, col_ticks_label)
        #ax.set_xlim(left = 0.0, right = 60)
        #ax.set_xlim(left = -10.0, right = 90)

    else:
        axcb.set_label(col_label)#)(r'Confirmation bias, $\theta$')
        ax.set_xlabel(row_label)#(r"Number of behaviours per agent, M")
        #ax.set_xticks(row_ticks_pos)
        #ax.set_xticklabels(row_ticks_label)
        #print("x ticks", row_label, row_ticks_pos, row_ticks_label)
        #ax.set_xlim(left = 1.0)
        #ax.set_xlim(left = 0.0, right = 2.0)

    #f = plotName + "/multi_line_matrix_plot_%s_%s" % (Y_param, col_axis_x)
    #plotName = fileName + "/Plots"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    #fig.savefig(f + ".png", dpi=dpi_save, format="png")   

def main(expenditure_list, carbon_price_list,params_dict):
    params_list = gen_params_dict(params_dict,expenditure_list, carbon_price_list, "B", "tau")
    
    data_emissions, data_low_carbon_prop = multi_emissions_low_carbon(params_list)

    data_emissions = data_emissions.reshape((len(expenditure_list),len(carbon_price_list)))
    data_low_carbon_prop = data_low_carbon_prop.reshape((len(expenditure_list),len(carbon_price_list)))
    
    #print("data_emissions",data_emissions.shape)
    row_label = "expenditure"#row_dict["title"]#r"Attitude Beta parameters, $(a,b)$"#r"Number of behaviours per agent, M"
    col_label = "Carbon price"#col_dict["title"]#r'Confirmation bias, $\theta$'

    quit()

    multi_line_matrix_plot(data_emissions, carbon_price_list, expenditure_list, get_cmap("plasma"), 0, col_label, row_label, "Emissions")#y_ticks_pos, y_ticks_label
    multi_line_matrix_plot(data_low_carbon_prop, carbon_price_list, expenditure_list, get_cmap("plasma"), 0, col_label, row_label, "Low carbon proportion")#y_ticks_pos, y_ticks_label

    
    plt.show()

# 2 is flying, 1 is food
if __name__ == "__main__":
    params_dict = {
        "a_1" : 0.2,
        "a_2" : 0.8,#greater preference for flying over food
        "P_H1" : 0.8,
        "P_H2" : 0.8,
        "P_L1" : 1.0,#unitary price
        "P_L2" : 1.0,#unitary price
        "A_1" : 0.5,#SAME PREFERENCE#high green preference of low carbon food/high carbon
        "A_2" : 0.5,#SAME PREFERENCE#weak preferecen for low carbon airtransport? 
        "sigma_1" : 10,#highish substitutability of low carbon food/energy
        "sigma_2" : 1.1,#low substitutability of flying
        "nu" : 5,#high substitutability between the branches?
        "B" : 10,
        "u_1" : 0.2,#need of food
        "u_2" : 0,#no need for flying
        "H1_0" : 1,
        "tau" : 0#carbon tax
    }
    expenditure_list = np.logspace(-1,1,4)
    print("expenditure_list",expenditure_list)
    carbon_price_list = np.linspace(0.0,1.0,3)# for low values of carbon tax it works

    main(expenditure_list, carbon_price_list,params_dict)