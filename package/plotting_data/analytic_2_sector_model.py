import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from package.resources.utility import produce_name_datetime, save_object, createFolder, load_object
from sympy import symbols, diff, simplify, lambdify, print_latex, And, solve
from scipy.optimize import brentq
from matplotlib.cm import get_cmap
#####################################
#PLOT 1D stuff for 1st figure 

def gen_1d_data(variable_parameters,parameters):
    variable_parameters["values"] = np.linspace(variable_parameters["min"], variable_parameters["max"], variable_parameters["reps"])

    parameters[variable_parameters["property"]] = variable_parameters["values"]

    return variable_parameters, parameters

def set_up_data_1D(root = "2_sector_model", init_params = 4, scenario = 1):
    """
    Scenario 1: is that sector 1 is a basic good with high subsititutability(FOOD), similar prices between both goods but a minimum of the high carbon required. But there is low preference for this sector
    Sector 2 has low substitutability, but more attractive (LONG DISTANCE TRAVEL), also the High carbon base price is much lower, howeer there is no minimum required
    - Assume individuals are indifferent to the environment
    """
    #############################################
    match init_params:
        case 4:
            parameters_1 = {
                "A1": 0.5,
                "A2": 0.5,
                "tau1": 0,
                "tau2": 0,
                "a1": 0.5,
                "a2": 0.5,
                "sigma1": 2,
                "sigma2": 2,
                "nu": 2,
                "PL1": 1,
                "PL2": 1,
                "PBH1": 1,
                "PBH2": 1,
                "h1": 0,
                "h2": 0,
                "B": 1,
                "gamma1":1,
                "gamma2":1
            }

    ##################################################
    match scenario:
        case 1:
            variable_parameters = {
                "property": "tau1",
                "min":0,
                "max": 1,
                "reps": 1000,
                "title": r'Sector 1 carbon price, $\tau_1$'
            }

    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    variable_parameters, parameters_run = gen_1d_data(variable_parameters, parameters_1)
    return fileName, variable_parameters, parameters_run, scenario, init_params

def calc_emissions_and_derivative(parameters):#THIS IS THE ONE I USE FOR RUNS!!!! FIGURE 1?
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
    B = parameters["B"] 
    tau1 = parameters["tau1"]
    gamma1 = parameters["gamma1"]
    gamma2 = parameters["gamma2"]

    # Define BD equation
    BD = B - h1 * (PBH1 + gamma1*tau1) - h2 * (PBH2 + gamma2*tau2)

    # Define Omega1 equation
    Omega1 = ((PBH1 + gamma1*tau1) * A1 / (PL1 * (1 - A1)))**sigma1

    # Define Omega2 equation
    Omega2 = ((PBH2 + tau2) * A2 / (PL2 * (1 - A2)))**sigma2

    # Define chi1 equation
    chi1 = ((a1 / (PBH1 + gamma1*tau1)) * ((A1 * Omega1**((sigma1 - 1) / sigma1)) + (1 - A1))**(((nu - 1) * sigma1) / (nu * (sigma1 - 1))))**nu

    # Define chi2 equation
    chi2 = ((a2 / (PBH2 + gamma2*tau2)) * ((A2 * Omega2**((sigma2 - 1) / sigma2)) + (1 - A2))**(((nu - 1) * sigma2) / (nu * (sigma2 - 1))))**nu

    # Define Z equation
    Z = (chi1 * (Omega1 * PL1 + PBH1 + gamma1*tau1)) + (chi2* (Omega2 * PL2 + PBH2 + gamma2*tau2))

    H1 = (BD*chi1) / Z + h1
    H2 = (BD*chi2) / Z + h2
    L1 = Omega1*H1
    L2 = Omega2*H2

    T = H1+ L1 + H2 + L2
    prop_sector_1 = (H1+L1)/T
    prop_H1 = H1/(H1+ L1)


    # Define EF equation with BD substituted
    EF1 = (BD * gamma1*chi1) / Z + gamma1*h1
    EF2 = (BD * gamma2*chi2) / Z + gamma2*h2
    EF = (BD * (gamma1*chi1 + gamma2*chi2) / Z) + gamma1*h1 + gamma2*h2

    return EF, EF1, EF2,  prop_sector_1, prop_H1, T

def calc_emissions_1_sector(parameters):
    A1 = parameters["A1"]
    a1 = parameters["a1"]
    sigma1 = parameters["sigma1"]
    nu = parameters["nu"]
    PL1 = parameters["PL1"]
    PBH1 = parameters["PBH1"]
    h1 = parameters["h1"]
    B = parameters["B"] 
    tau1 = parameters["tau1"]
    gamma1 = parameters["gamma1"]

    # Define BD equation
    BD = B - h1 * (PBH1 + gamma1*tau1)

    # Define Omega1 equation
    Omega1 = ((PBH1 + gamma1*tau1) * A1 / (PL1 * (1 - A1)))**sigma1

    # Define chi1 equation
    chi1 = (a1 / (PBH1 + gamma1*tau1)) * ((A1 * Omega1**((sigma1 - 1) / sigma1)) + (1 - A1))**(((nu - 1) * sigma1) / (nu * (sigma1 - 1)))

    # Define Z equation
    Z = (chi1**nu * (Omega1 * PL1 + PBH1 + gamma1*tau1))

    # Define EF equation with BD substituted
    EF1 = (BD * gamma1*chi1**nu) / Z + gamma1*h1

    return EF1

def plot_sector_emissions_rebound_line(fileName,variable_parameters, data_E_F, data_E_F_1, data_E_F_2, scenario, parameters):
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(7, 7),constrained_layout = True)

    var1_values = variable_parameters["values"]
    var1_title = variable_parameters["title"]

    #CALCULATE WHAT THE EMISSIONS WOULD BE IF SECTOR 2 DIDNT EXIST
    parameters["a1"] = 1#NEED TO CHANGE THIS ONE
    data_E_F_1sector = calc_emissions_1_sector(parameters)
    print("HELLO")
    
    # Plot E_F
    ax.plot(var1_values, data_E_F, label = r'Emissions both sectors, $E_F$')
    ax.plot(var1_values, data_E_F_1, label = r'Emissions sector 1, $E_{F,1}$')
    ax.plot(var1_values, data_E_F_2, label = r'Emissions sector 2, $E_{F,2}$')
    ax.plot(var1_values, data_E_F_1sector, label = r'Emissions no sector 2, $E^{*}_{F}$', ls="--")
    ax.set_xlabel(var1_title)
    ax.set_ylabel(r'Emissions, $E$')
    ax.legend()

    #plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/sector_emisisons_with_line" + variable_parameters["property"] 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def run_plots_1D(root,LOAD, init_params, scenario, LOAD_filename = "filename"):
    
    if LOAD:
        print("LOADED D")
        fileName = LOAD_filename
        print("fileName", fileName)
        parameters_run = load_object(fileName + "/Data","parameters_run")
        variable_parameters = load_object(fileName + "/Data","variable_parameters")
        data_E_F = load_object(fileName + "/Data","data_E_F")
        data_E_F_1 = load_object(fileName + "/Data","data_E_F_1")
        data_E_F_2 = load_object(fileName + "/Data","data_E_F_2")
        data_derv_E_F = load_object(fileName + "/Data","data_derv_E_F")
        data_EF_tau1_tau2 = load_object(fileName + "/Data","data_EF_tau1_tau2")
        scenario = load_object(fileName + "/Data","scenario")
    else:
        fileName, variable_parameters, parameters_run, scenario, init_params = set_up_data_1D(root, init_params, scenario)
        
        #data_E_F_1, data_derv_E_F_1 = calculate_data_alt(parameters_run)
        data_E_F, data_E_F_1, data_E_F_2, prop_sec1 ,prop_H1  = calc_emissions_and_derivative(parameters_run)

        createFolder(fileName)

        save_object(variable_parameters, fileName + "/Data", "variable_parameters")
        save_object(parameters_run, fileName + "/Data", "parameters_run")
        save_object(data_derv_E_F, fileName + "/Data", "data_derv_E_F")
        save_object(data_E_F_1, fileName + "/Data", "data_E_F_1")
        save_object(data_E_F_2, fileName + "/Data", "data_E_F_2")
        save_object(data_E_F_1, fileName + "/Data", "data_E_F")
        save_object(scenario, fileName + "/Data", "scenario")
        save_object(init_params, fileName + "/Data", "init_params")
    
    levels = 20
    #if scenario not in (2,3):
    #    plot_contours_gen(fileName,variable_parameters, data_E_F, data_derv_E_F,levels, 1, scenario)
    #plot_line_with_colorbar(fileName,variable_parameters, data_E_F, data_derv_E_F, 0, scenario)
    #plot_line_with_colorbar(fileName, variable_parameters, data_E_F, data_derv_E_F, 1, scenario)

    #plot_sector_emissions(fileName,variable_parameters, data_E_F, data_E_F_1, data_E_F_2, scenario)
    plot_sector_emissions_rebound_line(fileName,variable_parameters, data_E_F, data_E_F_1, data_E_F_2, scenario, parameters_run)
    #plot_contours(fileName, "Scenario 1",B_values, tau_values, data_E_F_1, data_derv_E_F_1, axis_val)
    #plot_contours(fileName, "Scenario 2",B_values, tau_values, data_E_F_2, data_derv_E_F_2, axis_val)
    #plot_contours_2_scen(fileName, B_values, tau_values, data_E_F_1, data_derv_E_F_1, data_E_F_2, data_derv_E_F_2, axis_val)

    plt.show()

######################################################################################################
    #STUFF FOR SECOND PLOT TAX REBOUND
def calc_emissions_tax_rebound():

    #THE ONE USE IN GRAPHS
    subs_dict = {#BASE PARAMS
        "A1": 0.5,
        "A2": 0.5,
        "tau2I": 0,
        "a1": 0.5,
        "a2": 0.5,
        "sigma1": 2,
        "sigma2": 2,
        "PL1": 1,
        "PL2": 1,
        "PBH1": 1,
        "PBH2": 1,
        "h1": 0,
        "h2": 0,
        "B": 1,
        "gamma1": 1,#0.8,
        "gamma2": 1,#1.1
    }

    #NOTE THAT YOU CANT SET PARAMETERS LATER, SO IF THEY ARE INCLUDED IN THIS LIST IT SFIXE D

    tau_C_sym, tau1_I_sym, tau2_I_sym, B_sym, nu_sym, h1_sym, h2_sym, PBH1_sym, PBH2_sym, A1_sym, A2_sym, PL1_sym, PL2_sym, sigma1_sym, sigma2_sym, a1_sym, a2_sym, gamma1_sym, gamma2_sym = symbols('tauC tau1I tau2I B nu h1 h2 PBH1 PBH2 A1 A2 PL1 PL2 sigma1 sigma2 a1 a2 gamma1 gamma2')
    
    # Define COMPLETE TAX STUFF
    BD_C_sym = B_sym - h1_sym * (PBH1_sym + gamma1_sym*tau_C_sym) - h2_sym * (PBH2_sym + gamma2_sym*tau_C_sym)
    Omega1_C_sym = ((PBH1_sym + gamma1_sym*tau_C_sym) * A1_sym / (PL1_sym * (1 - A1_sym)))**sigma1_sym
    Omega2_C_sym = ((PBH2_sym + gamma2_sym*tau_C_sym) * A2_sym / (PL2_sym * (1 - A2_sym)))**sigma2_sym
    chi1_C_sym = (a1_sym / (PBH1_sym + gamma1_sym*tau_C_sym)) * ((A1_sym * Omega1_C_sym**((sigma1_sym - 1) / sigma1_sym)) + (1 - A1_sym))**(((nu_sym - 1) * sigma1_sym) / (nu_sym * (sigma1_sym - 1)))
    chi2_C_sym = (a2_sym / (PBH2_sym + gamma2_sym*tau_C_sym)) * ((A2_sym * Omega2_C_sym**((sigma2_sym - 1) / sigma2_sym)) + (1 - A2_sym))**(((nu_sym - 1) * sigma2_sym) / (nu_sym * (sigma2_sym - 1)))
    Z_C_sym = (chi1_C_sym**nu_sym * (Omega1_C_sym * PL1_sym + PBH1_sym + gamma1_sym*tau_C_sym)) + (chi2_C_sym**nu_sym * (Omega2_C_sym* PL2_sym + PBH2_sym + gamma2_sym*tau_C_sym))

    # Define INCOMPLETE TAX STUFF
    BD_I_sym = B_sym - h1_sym * (PBH1_sym + gamma1_sym*tau1_I_sym) - h2_sym * (PBH2_sym + gamma2_sym*tau2_I_sym)
    Omega1_I_sym = ((PBH1_sym + gamma1_sym*tau1_I_sym) * A1_sym / (PL1_sym * (1 - A1_sym)))**sigma1_sym
    Omega2_I_sym = ((PBH2_sym + gamma2_sym*tau2_I_sym) * A2_sym / (PL2_sym * (1 - A2_sym)))**sigma2_sym
    chi1_I_sym = (a1_sym / (PBH1_sym + gamma1_sym*tau1_I_sym)) * ((A1_sym * Omega1_I_sym**((sigma1_sym - 1) / sigma1_sym)) + (1 - A1_sym))**(((nu_sym - 1) * sigma1_sym) / (nu_sym * (sigma1_sym - 1)))
    chi2_I_sym = (a2_sym / (PBH2_sym + gamma2_sym*tau2_I_sym)) * ((A2_sym * Omega2_I_sym**((sigma2_sym - 1) / sigma2_sym)) + (1 - A2_sym))**(((nu_sym - 1) * sigma2_sym) / (nu_sym * (sigma2_sym - 1)))
    Z_I_sym = (chi1_I_sym**nu_sym * (Omega1_I_sym * PL1_sym + PBH1_sym + gamma1_sym*tau1_I_sym)) + (chi2_I_sym**nu_sym * (Omega2_I_sym * PL2_sym + PBH2_sym + gamma2_sym*tau2_I_sym))

    EF_C = (BD_C_sym * (gamma1_sym*chi1_C_sym**nu_sym + gamma2_sym*chi2_C_sym**nu_sym) / Z_C_sym) + gamma1_sym*h1_sym + gamma2_sym*h2_sym
    EF_I = (BD_I_sym * (gamma1_sym*chi1_I_sym**nu_sym + gamma2_sym*chi2_I_sym**nu_sym) / Z_I_sym) + gamma1_sym*h1_sym + gamma2_sym*h2_sym

    equation = EF_C - EF_I

    equation_simplified = equation.subs(subs_dict)
    print("SIMPLE EQUATION", equation_simplified)

    tau_C_solution_no_nu_sub = solve(equation_simplified, tau_C_sym, positive=True)
    print("SOLUTION FOUND", tau_C_solution_no_nu_sub)
    print(len(tau_C_solution_no_nu_sub))
    
    
    #######################################################################################################################################################
    root = "nu_tau_ratio"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    createFolder(fileName)


    #save_object(tau_C_func_1, fileName + "/Data", "tau_C_func_1")
    save_object(subs_dict, fileName + "/Data", "subs_dict")

    #######################################################################################################################################################
    fig_together, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), constrained_layout=True)

    ax1 = axes[0]
    ax2 = axes[1]
    #ax3 = axes[2]
    
    #I CANT BE BOTHERED TO RE-WRITE THIS ATM

    tau_C_func_1 = lambdify((nu_sym,tau1_I_sym), tau_C_solution_no_nu_sub[0], 'numpy')#FOR SOME REASON YOU HAVE TO GET THE SECOND ONE???

    #NU VARS

    data_tau1_tau = []
    data_tauC = []

    nu_values_lin_tau = np.logspace(np.log10(1.01), 1, 3)
    tau1_I_values_lin = np.linspace(0.01, 1, 1000)
    for nu in nu_values_lin_tau:
        tau_C_values_1 = tau_C_func_1(nu, tau1_I_values_lin)
        data_tau1_tau.append( tau1_I_values_lin/tau_C_values_1)#1 -
        data_tauC.append(tau_C_values_1)

    save_object(nu_values_lin_tau , fileName + "/Data", "nu_values_lin_tau")
    save_object(tau1_I_values_lin , fileName + "/Data", "tau1_I_values_lin ")
    save_object(data_tau1_tau, fileName + "/Data", "data_tau1_tau")

    #fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5),constrained_layout = True)
    
    for i,nu in enumerate(nu_values_lin_tau):
        ax2.plot(tau1_I_values_lin, data_tau1_tau[i], label= "$\\nu = %s$" % (round(nu,3)))
    #ax2.plot(nu_values_lin, tau_C_values_1, label= "Positive solutions")
    #ax.plot(nu_values, tau_C_values_0, label= "Negative solutions")
    ax2.plot(tau1_I_values_lin, [1]*len(tau1_I_values_lin) , color = "black",ls = "--", alpha = 0.5)
    ax2.set_xlabel(r"Sector 1 incomplete carbon tax, $\tau_I$ ")
    ax2.set_ylabel(r"Rebound tax multiplier, $M_R = \frac{\tau_I}{\tau_C}$")#1 - 
    #ax2.set_title('linear')
    #ax2.legend()
    ax2.grid()

    #COMPLETE CARBON TAX
    for i,nu in enumerate(nu_values_lin_tau):
        ax1.plot(tau1_I_values_lin, data_tauC[i], label= "$\\nu = %s$" % (round(nu,3)))


    ax1.plot(tau1_I_values_lin,tau1_I_values_lin , color = "black",ls = "--", alpha = 0.5, label = "No multiplier")

    ax1.set_xlabel(r"Sector 1 incomplete carbon tax, $\tau_I$ ")
    ax1.set_ylabel(r"Complete carbon tax,  $\tau_C$")#1 - 
    #ax2.set_title('linear')
    ax1.legend()
    ax1.grid()



    # TAU VARS
    """
    data_tau1_nu_tau = []

    nu_values_lin_nu_tau = np.linspace(1.01, 10, 1000)
    tau1_I_values_lin_nu_tau = np.linspace(0.01, 1, 3)
    for tau_I in tau1_I_values_lin_nu_tau:
        tau_C_values_1 = tau_C_func_1(nu_values_lin_nu_tau, tau_I)
        data_tau1_nu_tau.append(tau_I/tau_C_values_1)#1 -

    save_object(nu_values_lin_nu_tau , fileName + "/Data", "nu_values_lin_nu_tau")
    save_object(tau1_I_values_lin_nu_tau , fileName + "/Data", "tau1_I_values_lin_nu_tau")
    save_object(data_tau1_nu_tau, fileName + "/Data", "data_tau1_nu_tau")

    #fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5),constrained_layout = True)
    
    for i,tau in enumerate(tau1_I_values_lin_nu_tau):
        ax3.plot(nu_values_lin_nu_tau , data_tau1_nu_tau[i], label= "$\\tau_I = %s$" % (round(tau,3)))
    ax3.set_xlabel(r"Inter-sector substitutability, $\nu$")
    ax3.set_ylabel(r"Rebound tax multiplier, $M_R = \frac{\tau_I}{\tau_C}$")#1 - 
    ax3.legend()
    ax3.grid()
    
    plt.tight_layout()


    """
    #######################################################################################################################################################
    """
    #ALT SOLUTION

    
    tau_C_func_1_alt = lambdify((nu_sym,tau1_I_sym), tau_C_solution_no_nu_sub[1], 'numpy')#FOR SOME REASON YOU HAVE TO GET THE SECOND ONE???

    #NU VARS

    data_tau1_tau_alt = []

    nu_values_lin_tau_alt = np.logspace(np.log10(1.01), 2, 5)
    tau1_I_values_lin_alt = np.linspace(0.1, 1, 1000)
    for nu_alt in nu_values_lin_tau_alt:
        tau_C_values_1_alt = tau_C_func_1_alt(nu_alt, tau1_I_values_lin_alt)
        data_tau1_tau_alt.append( tau1_I_values_lin_alt/tau_C_values_1_alt)#1 -
    
    fig_alt, ax_alt = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for i, nu_alt in enumerate(nu_values_lin_tau_alt):
        ax_alt.plot(tau1_I_values_lin_alt, data_tau1_tau_alt[i], label= "$\\nu = %s$" % (round(nu_alt,3)))
    ax_alt.set_xlabel(r"Sector 1 incomplete carbon tax,$\tau_I$ ")
    ax_alt.set_ylabel(r"Rebound tax multiplier, $M_R = \frac{\tau_I}{\tau_C}$")#1 - 
    #ax2.set_title('linear')
    ax_alt.legend()
    ax_alt.grid()
    """
    
    plotName = fileName + "/Plots"
    f = plotName + "/together_incomplete"
    fig_together.savefig(f + ".eps", dpi=600, format="eps")
    fig_together.savefig(f + ".png", dpi=600, format="png")

    plt.show()

def root_emmissions_function(tau_C, *args):

    tau_I, v = args

    numerator_I = 4**(1-v) * (tau_I+2)**(2*(v-1)) + (1+tau_I)**v
    denominator_I = 4**(1-v) * (tau_I+2)**(2*v-1) * (1+tau_I) + 2*(1+tau_I)**v
    E_I = numerator_I / denominator_I
    E_C =  1 / ((1 + tau_C) * (2 + tau_C))
    convergence_val = E_I - E_C
    return convergence_val

def calc_emissions(tau_C, tau_I, v):
    numerator_I = 4**(1-v) * (tau_I+2)**(2*(v-1)) + (1+tau_I)**v
    denominator_I = 4**(1-v) * (tau_I+2)**(2*v-1) * (1+tau_I) + 2*(1+tau_I)**v
    E_I = numerator_I / denominator_I
    E_C =  1 / ((1 + tau_C) * (2 + tau_C))
    return E_I, E_C


def calc_ratio_complete_incomplete():
    
    tau_I_list = np.linspace(0,1,1000)
    v_list = [1.01,5,30]

    tau_C_data = []
    E_I_data = []
    for v in v_list:
        tau_C_list = []
        E_I_list = []
        E_C_list = []
        for tau_I in tau_I_list:
            E_I, E_C = calc_emissions(tau_I,tau_I, v)
            E_I_list.append(E_I)
            E_C_list.append(E_C)
            root = brentq(f = root_emmissions_function, a = 0, b = 10000, args=(tau_I,v), maxiter= 100)
            tau_C_list.append(root)
        tau_C_data.append(tau_C_list)
        E_I_data.append(E_I_list)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), constrained_layout=True)

    for i, tau_C_v in enumerate(tau_C_data):
        ax.plot(tau_I_list, tau_C_v, label= "$\\nu = %s$" % (v_list[i]))

    ax.set_xlabel(r"Sector 1 incomplete carbon tax, $\tau_I$ ")
    ax.set_ylabel(r"Complete carbon tax,  $\tau_C$")#1 - 
    #ax2.set_title('linear')
    ax.legend()
    ax.grid()

    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), constrained_layout=True)

    for i, E_I in enumerate(E_I_data):
        ax2.plot(tau_I_list, E_I, label= "Incomplete coverage, $\\nu = %s$" % (v_list[i]))

    ax2.plot(tau_I_list, E_C_list, label= "Complete coverage")

    ax2.set_xlabel(r"Carbon tax in complete and incomplete coverage, $\tau$")
    ax2.set_ylabel(r"Emmisions flow, $E_F$")#1 - 
    #ax2.set_title('linear')
    ax2.legend()
    ax2.grid()

    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), constrained_layout=True)

    for i, E_I in enumerate(E_I_data):
        ax3.plot(E_I, tau_I_list, label= "Incomplete coverage, $\\nu = %s$" % (v_list[i]))

    ax3.plot(E_C_list,tau_I_list, label= "Complete coverage")

    ax3.set_ylabel(r"Carbon tax in complete and incomplete coverage, $\tau$")
    ax3.set_xlabel(r"Emmisions flow, $E_F$")#1 - 
    #ax2.set_title('linear')
    ax3.legend()
    ax3.grid()

    #######################################################################################
    # Values of tau
    tau_values_5 = np.linspace(0, 1, 1000)

    # Values of x (nu)
    nu_values_5 = [1.01,5,30]  # Adjust range and number of points as needed

    def new_sector1_consumption_proportion(tau1, nu):
        numerator = (1 + tau1)**(-nu) * (2 + tau1)**(2*(nu - 1)) * (1 + (1 + tau1)**2)
        denominator = numerator + 2**(2*nu - 1)
        return numerator / denominator

    # Plotting
    fig5, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), constrained_layout=True)

    for nu in nu_values_5:
        sector1_prop = [new_sector1_consumption_proportion(tau,nu) for tau in tau_values_5]
        ax5.plot(tau_values_5, sector1_prop, label="Sector 1 consumption propotion, $\\nu$ = %s" % (nu))

    Omega_prop = [1/(1+(1+tau**2)) for tau in tau_values_5]
    ax5.plot(tau_values_5, Omega_prop, label="Sector 1 High consumption propotion",linestyle="--" )

    ax5.set_xlabel('Carbon tax, $\\tau$')
    ax5.set_ylabel('Proportion')
    ax5.legend()

    root = "E_root_nu_tau_ratio"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    createFolder(fileName)

    plotName = fileName + "/Plots"
    f = plotName + "/Emissions"
    fig2.savefig(f + ".eps", dpi=600, format="eps")
    fig2.savefig(f + ".png", dpi=600, format="png")

    f = plotName + "/ratio"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

    f = plotName + "/ratio_sector_1_consumption_low_carb"
    fig5.savefig(f + ".eps", dpi=600, format="eps")
    fig5.savefig(f + ".png", dpi=600, format="png")

    plt.show()

    return tau_C_list

def A_nu_tau_effect():
    # Values of tau
    tau_values = np.linspace(0, 1, 1000)
    nu_values = [1.01,5,30]  # Adjust range and number of points as needed
    A_1_values = [0.4,0.5,0.6]

    #def calc_emissions
    parameters_dict = {#BASE PARAMS
        "A2": 0.5,
        "tau2": 0,
        "a1": 0.5,
        "a2": 0.5,
        "sigma1": 2,
        "sigma2": 2,
        "PL1": 1,
        "PL2": 1,
        "PBH1": 1,
        "PBH2": 1,
        "h1": 0,
        "h2": 0,
        "B": 1,
        "gamma1": 1,
        "gamma2": 1,
    }

    data = np.zeros(shape = (len(A_1_values),len(nu_values), len(tau_values) ))
    data_sec1 = np.zeros(shape = (len(A_1_values),len(nu_values), len(tau_values) ))
    data_H1 = np.zeros(shape = (len(A_1_values),len(nu_values), len(tau_values) ))
    for i, A_1 in enumerate(A_1_values):
        parameters_dict["A1"] = A_1
        for j, nu in enumerate(nu_values):
            parameters_dict["nu"] = nu
            for k, tau in enumerate(tau_values):
                parameters_dict["tau1"] = tau
                #EF, manual_derv_EF_tau1, EF1, EF2
                data[i,j,k], _ , _ , data_sec1[i,j,k] ,data_H1[i,j,k] = calc_emissions_and_derivative(parameters_dict)

    fig, axes = plt.subplots(nrows=1, ncols=len(A_1_values), figsize=(10, 6), constrained_layout=True, sharey=True)

    for i, A_1 in enumerate(A_1_values):
        axes[i].set_title("Sector 1 low carbon preference, $A_1$ = %s" % (A_1))
        axes[i].set_xlabel('Carbon tax, $\\tau$')
        for j, nu in enumerate(nu_values):
            axes[i].plot(tau_values, data[i][j], label="$\\nu$ = %s" % (nu))
            

    axes[0].set_ylabel('Emissions flow, $E_F$')
    axes[-1].legend()

    fig1, axes1 = plt.subplots(nrows=1, ncols=len(A_1_values), figsize=(10, 6), constrained_layout=True, sharey=True)

    for i, A_1 in enumerate(A_1_values):
        axes1[i].set_title("Sector 1 low carbon preference, $A_1$ = %s" % (A_1))
        axes1[i].set_xlabel('Carbon tax, $\\tau$')
        for j, nu in enumerate(nu_values):
            axes1[i].plot(tau_values, data_sec1[i][j], label="$\\nu$ = %s" % (nu))
            
    axes1[0].set_ylabel('Proportion of total consumption sector 1')
    axes1[-1].legend()

    fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), constrained_layout=True, sharey=True)

    for i, A_1 in enumerate(A_1_values):
        axes2.plot(tau_values, data_sec1[i][0], label="Sector 1 low carbon preference, $A_1$ = %s" % (A_1))
    axes2.set_xlabel('Carbon tax, $\\tau$')   
    axes2.set_ylabel("Proportion of sector 1 consumption low carbon")
    axes2.legend()

    ###############################################################################

    parameters_dict_a = {#BASE PARAMS
        "A1": 0.5,
        "A2": 0.5,
        "tau2": 0,
        "sigma1": 2,
        "sigma2": 2,
        "PL1": 1,
        "PL2": 1,
        "PBH1": 1,
        "PBH2": 1,
        "h1": 0,
        "h2": 0,
        "B": 1,
        "gamma1": 1,
        "gamma2": 1,
    }

    a_values = [0.4,0.5,0.6]
    data_a = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    for i, a in enumerate(a_values):
        parameters_dict_a["a1"] = a
        parameters_dict_a["a2"] = 1-a
        for j, nu in enumerate(nu_values):
            parameters_dict_a["nu"] = nu
            for k, tau in enumerate(tau_values):
                parameters_dict_a["tau1"] = tau
                #EF, manual_derv_EF_tau1, EF1, EF2
                data_a[i,j,k], _ , _ , _, _ = calc_emissions_and_derivative(parameters_dict_a)

    fig3, axes3 = plt.subplots(nrows=1, ncols=len(a_values), figsize=(10, 6), constrained_layout=True, sharey=True)

    for i, a in enumerate(a_values):
        axes3[i].set_title("Preference for Sector 1 goods, $a$ = %s" % (a))
        axes3[i].set_xlabel('Carbon tax, $\\tau$')
        for j, nu in enumerate(nu_values):
            axes3[i].plot(tau_values, data_a[i][j], label="$\\nu$ = %s" % (nu))
            
    axes3[0].set_ylabel('Emissions flow, $E_F$')
    axes3[-1].legend()

    ###############################################################################
    
    root = "A_nu_tau"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    createFolder(fileName)

    plotName = fileName + "/Plots"
    f = plotName + "/Emissions"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

    plotName = fileName + "/Plots"
    f = plotName + "/prop_sec1"
    fig1.savefig(f + ".eps", dpi=600, format="eps")
    fig1.savefig(f + ".png", dpi=600, format="png")

    plotName = fileName + "/Plots"
    f = plotName + "/prop_h1"
    fig2.savefig(f + ".eps", dpi=600, format="eps")
    fig2.savefig(f + ".png", dpi=600, format="png")

    plotName = fileName + "/Plots"
    f = plotName + "/prop_sec1_a"
    fig3.savefig(f + ".eps", dpi=600, format="eps")
    fig3.savefig(f + ".png", dpi=600, format="png")

    plt.show()

def a_nu_tau_effect():
        # Values of tau
    tau_values = np.linspace(0, 1, 1000)
    nu_values = [1.01,5,30]  # Adjust range and number of points as needed
    a_values = [0.4,0.5,0.6]

    ######################################
    #Colours
    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list
    ####################################


    parameters_dict_a = {#BASE PARAMS
        "A1": 0.5,
        "A2": 0.5,
        "tau2": 0,
        "sigma1": 2,
        "sigma2": 2,
        "PL1": 1,
        "PL2": 1,
        "PBH1": 1,
        "PBH2": 1,
        "h1": 0,
        "h2": 0,
        "B": 1,
        "gamma1": 1,
        "gamma2": 1,
    }
    line_width = 1.5
    
    data = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    data_1 = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    data_2 = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    data_sec1 = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    data_H1 = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    data_total = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    for i, a in enumerate(a_values):
        parameters_dict_a["a1"] = a
        parameters_dict_a["a2"] = 1-a
        for j, nu in enumerate(nu_values):
            parameters_dict_a["nu"] = nu
            for k, tau in enumerate(tau_values):
                parameters_dict_a["tau1"] = tau
                #EF, manual_derv_EF_tau1, EF1, EF2
                data[i,j,k], data_1[i,j,k] , data_2[i,j,k] , data_sec1[i,j,k] ,data_H1[i,j,k] , data_total[i,j,k]= calc_emissions_and_derivative(parameters_dict_a)

    ###############################################################################
    """
    fig3, axes3 = plt.subplots(nrows=1, ncols=len(a_values), figsize=(15, 6), constrained_layout=True, sharey=True)

    for i, a in enumerate(a_values):
        axes3[i].set_title("Preference for Sector 1 goods, $a$ = %s" % (a))
        axes3[i].set_xlabel('Carbon tax, $\\tau$')
        for j, nu in enumerate(nu_values):
            axes3[i].plot(tau_values, data[i][j], label="$\\nu$ = %s" % (nu), color = colors_scenarios[j], linewidth= line_width)
    axes3[0].set_ylabel('Emissions flow, $E_F$')
    axes3[-1].legend()

    

    fig1, axes1 = plt.subplots(nrows=1, ncols=len(a_values), figsize=(15, 6), constrained_layout=True, sharey=True)

    for i, a in enumerate(a_values):
        axes1[i].set_title("Preference for Sector 1 goods, $a$ = %s" % (a))
        axes1[i].set_xlabel('Carbon tax, $\\tau$')
        for j, nu in enumerate(nu_values):
            axes1[i].plot(tau_values, data_sec1[i][j], label="$\\nu$ = %s" % (nu))
            
    axes1[0].set_ylabel('Proportion of total consumption sector 1')
    axes1[-1].legend()

    fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), constrained_layout=True, sharey=True)

    for i, a in enumerate(a_values):
        axes2.plot(tau_values, data_sec1[i][0], label="Preference for Sector 1 goods, $a$ = %s" % (a))
    axes2.set_xlabel('Carbon tax, $\\tau$')   
    axes2.set_ylabel("Proportion of sector 1 consumption low carbon")
    axes2.legend()
    
    ##################################################################################################
    
    fig4, axes4 = plt.subplots(nrows=1, ncols=len(a_values), figsize=(15, 6), constrained_layout=True, sharey=True)

    colour_list = [ "red", "blue", "green", "yellow", "purple", "orange", "white", "black" ]


    for i, a in enumerate(a_values):
        axes4[i].set_title("Preference for Sector 1 goods, $a$ = %s" % (a))
        axes4[i].set_xlabel('Carbon tax, $\\tau$')
        for j, nu in enumerate(nu_values):
            axes4[i].plot(tau_values, data_1[i][j], label="Sector 1, $\\nu$ = %s" % (nu), linestyle = "-.", color = colors_scenarios[j], linewidth= line_width)
            axes4[i].plot(tau_values, data_2[i][j], label="Sector 2, $\\nu$ = %s" % (nu), linestyle = "--", color = colors_scenarios[j], linewidth= line_width)
            
    axes4[0].set_ylabel('Sectoral emissions flow, $E_{1,2}$')
    axes4[-1].legend()
    """
    ################################################################################################

    # ADD IN SOLID LINE OF THE COMPLETE COVERAGE CASE
    complete_data = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    complete_data_1 = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    complete_data_2 = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    complete_data_sec1 = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    complete_data_H1 = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))
    complete_data_total = np.zeros(shape = (len(a_values),len(nu_values), len(tau_values) ))

    for i, a in enumerate(a_values):
        parameters_dict_a["a1"] = a
        parameters_dict_a["a2"] = 1-a
        for j, nu in enumerate(nu_values):
            parameters_dict_a["nu"] = nu
            for k, tau in enumerate(tau_values):
                parameters_dict_a["tau1"] = tau
                parameters_dict_a["tau2"] = tau
                #EF, manual_derv_EF_tau1, EF1, EF2
                complete_data[i,j,k], complete_data_1[i,j,k] , complete_data_2[i,j,k] , complete_data_sec1[i,j,k] ,complete_data_H1[i,j,k], complete_data_total[i,j,k] = calc_emissions_and_derivative(parameters_dict_a)
    """
    fig5, axes5 = plt.subplots(nrows=1, ncols=len(a_values), figsize=(15, 6), constrained_layout=True, sharey=True)

    colour_list = [ "red", "blue", "green", "yellow", "purple", "orange", "white", "black" ]

    for i, a in enumerate(a_values):
        axes5[i].set_title("Preference for Sector 1 goods, $a$ = %s" % (a))
        axes5[i].set_xlabel('Carbon tax, $\\tau$')
        for j, nu in enumerate(nu_values):
            axes5[i].plot(tau_values, complete_data_1[i][j], label="Sector 1, $\\nu$ = %s" % (nu), linestyle = "-.", color = colors_scenarios[j], linewidth= line_width)
            axes5[i].plot(tau_values, complete_data_2[i][j], label="Sector 2, $\\nu$ = %s" % (nu), linestyle = "--", color = colors_scenarios[j], linewidth= line_width)
            
    axes5[0].set_ylabel('Sectoral emissions flow, $E_{1,2}$')
    axes5[-1].legend()


    #NOW DO THE SECTORAL AND INTRA SECTORAL ON THE SAME PLOT

    fig6, axes6 = plt.subplots(nrows=1, ncols=len(a_values), figsize=(15, 6), constrained_layout=True, sharey=True)

    for i, a in enumerate(a_values):
        axes6[i].set_title("Preference for Sector 1 goods, $a$ = %s" % (a))
        axes6[i].set_xlabel('Carbon tax, $\\tau$')
        
        axes6[i].plot(tau_values, complete_data_H1[i][0], label="Sector 1 High carbon", linestyle = "solid", color = "black", linewidth= line_width)
        axes6[i].plot(tau_values, complete_data_sec1[i][j], label="Complete Sector 1 Total, $\\nu$ = %s" % (nu), linestyle = "solid", color = "orange", linewidth= line_width)
        for j, nu in enumerate(nu_values):
            axes6[i].plot(tau_values, data_sec1[i][j], label="Incomplete Sector 1 Total, $\\nu$ = %s" % (nu), linestyle = "--", color = colors_scenarios[j], linewidth= line_width)

    axes6[0].set_ylabel('Proportion')
    axes6[-1].legend()
    """

    #JOINT PLOT
    fig7, axes7 = plt.subplots(nrows=3, ncols=len(a_values), figsize=(12, 12), constrained_layout=True, sharey="row", sharex=True)

    for i, a in enumerate(a_values):
        
        axes7[0][i].set_title("Preference for Sector 1 goods, $a$ = %s" % (a))
        axes7[2][i].set_xlabel('Carbon tax, $\\tau$')
        
        axes7[2][i].plot(tau_values, complete_data_H1[i][0], label="Sector 1 High carbon", linestyle = "solid", color = "black", linewidth= line_width)
        #axes7[2][i].plot(tau_values, complete_data_sec1[i][j], label="Complete Sector 1 Total, $\\nu$ = %s" % (nu), linestyle = "solid", color = "orange", linewidth= line_width)
        for j, nu in enumerate(nu_values):
            axes7[0][i].plot(tau_values, data[i][j], label="$\\nu$ = %s" % (nu), color = colors_scenarios[j], linewidth= line_width)
            axes7[1][i].plot(tau_values, data_1[i][j], label="Sector 1, $\\nu$ = %s" % (nu), linestyle = "-.", color = colors_scenarios[j], linewidth= line_width)
            axes7[1][i].plot(tau_values, data_2[i][j], label="Sector 2, $\\nu$ = %s" % (nu), linestyle = "--", color = colors_scenarios[j], linewidth= line_width)
            axes7[2][i].plot(tau_values, data_sec1[i][j], label="Incomplete Sector 1 Total, $\\nu$ = %s" % (nu), linestyle = "--", color = colors_scenarios[j], linewidth= line_width)
        
    axes7[0][-1].legend()
    axes7[1][-1].legend()
    axes7[2][-1].legend()
    axes7[0][0].set_ylabel('Emissions flow, $E_F$')
    axes7[1][0].set_ylabel('Sectoral emissions flow, $E_{1,2}$')
    axes7[2][0].set_ylabel('Proportion')
    
    fig8, axes8 = plt.subplots(nrows=1, ncols=len(a_values), figsize=(15, 6), constrained_layout=True, sharey=True)

    for i, a in enumerate(a_values):
        axes8[i].set_title("Preference for Sector 1 goods, $a$ = %s" % (a))
        axes8[i].set_xlabel('Carbon tax, $\\tau$')
    
        for j, nu in enumerate(nu_values):
            axes8[i].plot(tau_values, data_total[i][j], label="$\\nu$ = %s" % (nu),  color = colors_scenarios[j], linewidth= line_width)

    axes8[0].set_ylabel('Total consumption')
    axes8[-1].legend()


    ###############################################################################
    
    root = "a_nu_tau"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    createFolder(fileName)

    """
    plotName = fileName + "/Plots"
    f = plotName + "/prop_sec1"
    fig1.savefig(f + ".eps", dpi=600, format="eps")
    fig1.savefig(f + ".png", dpi=600, format="png")

    plotName = fileName + "/Plots"
    f = plotName + "/prop_h1"
    fig2.savefig(f + ".eps", dpi=600, format="eps")
    fig2.savefig(f + ".png", dpi=600, format="png")
    
    

    plotName = fileName + "/Plots"
    f = plotName + "/Emissions"
    fig3.savefig(f + ".eps", dpi=600, format="eps")
    fig3.savefig(f + ".png", dpi=600, format="png")
    
    
    plotName = fileName + "/Plots"
    f = plotName + "/sec_Emissions"
    fig4.savefig(f + ".eps", dpi=600, format="eps")
    fig4.savefig(f + ".png", dpi=600, format="png")

    plotName = fileName + "/Plots"
    f = plotName + "/COMPLETEsec_Emissions"
    fig5.savefig(f + ".eps", dpi=600, format="eps")
    fig5.savefig(f + ".png", dpi=600, format="png")

    plotName = fileName + "/Plots"
    f = plotName + "/prop_consum"
    fig6.savefig(f + ".eps", dpi=600, format="eps")
    fig6.savefig(f + ".png", dpi=600, format="png")
    plt.show()
    """

    plotName = fileName + "/Plots"
    f = plotName + "/joint_plot"
    fig7.savefig(f + ".eps", dpi=600, format="eps")
    fig7.savefig(f + ".png", dpi=600, format="png")


    plotName = fileName + "/Plots"
    f = plotName + "/total"
    fig7.savefig(f + ".eps", dpi=600, format="eps")
    fig7.savefig(f + ".png", dpi=600, format="png")

    plt.show()

def main( type_run):

    if type_run == "plots_1D":
        run_plots_1D(root = "2_sector_model", LOAD = 0, init_params = 4, scenario = 1)#init_params = 8, scenario = 4
    elif type_run == "rebound_tax":
        calc_emissions_tax_rebound()
    elif type_run =="ratio":
        calc_ratio_complete_incomplete()
    elif type_run == "A_nu_tau":
        A_nu_tau_effect()
    elif type_run == "a_nu_tau":
        a_nu_tau_effect()
    else:
        raise ValueError("Wrong TYPE")

if __name__ == '__main__':
    type_run = "a_nu_tau"#"rebound_tax"# "plots_1D"
    main(type_run)
