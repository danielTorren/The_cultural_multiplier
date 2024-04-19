import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from package.resources.utility import produce_name_datetime, save_object, createFolder, load_object
from sympy import symbols, diff, simplify, lambdify, print_latex, And, solve

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
    chi1 = (a1 / (PBH1 + gamma1*tau1)) * ((A1 * Omega1**((sigma1 - 1) / sigma1)) + (1 - A1))**(((nu - 1) * sigma1) / (nu * (sigma1 - 1)))

    # Define chi2 equation
    chi2 = (a2 / (PBH2 + gamma2*tau2)) * ((A2 * Omega2**((sigma2 - 1) / sigma2)) + (1 - A2))**(((nu - 1) * sigma2) / (nu * (sigma2 - 1)))

    # Define Z equation
    Z = (chi1**nu * (Omega1 * PL1 + PBH1 + gamma1*tau1)) + (chi2**nu * (Omega2 * PL2 + PBH2 + gamma2*tau2))

    # Define EF equation with BD substituted
    EF1 = (BD * gamma1*chi1**nu) / Z + gamma1*h1
    EF2 = (BD * gamma2*chi2**nu) / Z + gamma2*h2
    EF = (BD * (gamma1*chi1**nu + gamma2*chi2**nu) / Z) + gamma1*h1 + gamma2*h2
    manual_derv_EF_tau1 = gamma1 * BD * Z**(-1) * nu * chi1**(nu - 1) * ((gamma1*chi1 / (PBH1 + gamma1*tau1)) * ((sigma1 * (nu - 1) * A1 * Omega1**((sigma1 - 1) / sigma1)) / (nu * (A1 * Omega1**((sigma1 - 1) / sigma1) + (1 - A1))) - 1)) - Z**(-1) * gamma1 * h1 * (gamma1 * chi1**nu + gamma2 * chi2**nu) - BD * Z**(-2) * (gamma1 * chi1**nu + gamma2 * chi2**nu) * (((Omega1 * PL1 + PBH1 + gamma1*tau1) * gamma1 * nu * chi1**nu) / (PBH1 + gamma1*tau1) * ((sigma1 * (nu - 1) * A1 * Omega1**((sigma1 - 1) / sigma1))/(nu * (A1 * Omega1**((sigma1 - 1) / sigma1) + (1 - A1))) - 1) + gamma1*chi1**nu * (PL1 * (sigma1 * Omega1) / (PBH1 + gamma1*tau1) + 1))

    return EF, manual_derv_EF_tau1, EF1, EF2

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
        data_E_F, data_derv_E_F, data_E_F_1, data_E_F_2  = calc_emissions_and_derivative(parameters_run)

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

def main( type_run):

    if type_run == "plots_1D":
        run_plots_1D(root = "2_sector_model", LOAD = 0, init_params = 4, scenario = 1)#init_params = 8, scenario = 4
    elif type_run == "rebound_tax":
        calc_emissions_tax_rebound()
    else:
        raise ValueError("Wrong TYPE")

if __name__ == '__main__':
    type_run = "rebound_tax"# "plots_1D"
    main(type_run)
