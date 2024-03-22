import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from package.resources.utility import produce_name_datetime, save_object, createFolder, load_object
from sympy import symbols, diff, simplify, lambdify, print_latex

##############################################################################################################
def calculate_data_alt(parameters):
    #print("Parameters",parameters)
    #quit()

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
    
    Omega1 = ((PBH1 + tau1) * A1) / (PL1 * (1 - A1)) ** sigma1
    Omega2 = ((PBH2 + tau2) * A2) / (PL2 * (1 - A2)) ** sigma2
    
    chi1 = (a1 / (PBH1 + tau1)) * ((A1 * Omega1 ** ((sigma1 - 1) / sigma1)) + (1 - A1)) ** ((nu - 1) * sigma1 / (nu * (sigma1 - 1)))
    chi2 = (a2 / (PBH2 + tau2)) * ((A2 * Omega2 ** ((sigma2 - 1) / sigma2)) + (1 - A2)) ** ((nu - 1) * sigma2 / (nu * (sigma2 - 1)))
    
    Z = (chi1 ** nu) * (Omega1 * PL1 + PBH1 + tau1) + (chi2 ** nu) * (Omega2 * PL2 + PBH2 + tau2)
    
    H1 = ((chi1 ** nu) * (B - h1 * (PBH1 + tau1) - h2 * (PBH2 + tau2))) / Z + h1
    H2 = ((chi2 ** nu) * (B - h1 * (PBH1 + tau1) - h2 * (PBH2 + tau2))) / Z + h2
    
    E_F = H1 + H2
    #E_F_alt = ((B - h1 * (PBH1 + tau1) - h2 * (PBH2 + tau2)) * (chi1 ** nu + chi2 ** nu)) / Z + h1 + h2
    #print("difff", E_F -E_F_alt )
    #quit()
    derivative_E_F = (-1 / Z ** 2) * (B - h1 * (PBH1 + tau1) - h2 * (PBH2 + tau2)) * (chi1 ** nu + chi2 ** nu) +  (1 / Z) * (-h1 * (chi1 ** nu + chi2 ** nu) + (B - h1 * (PBH1 + tau1) - h2 * (PBH2 + tau2)) * 
                                   (nu * chi1 ** nu * ((nu - 1) / nu * (A1 * Omega1 ** ((sigma1 - 1) / sigma1)) / 
                                   (A1 * Omega1 ** ((sigma1 - 1) / sigma1) - (1 - A1)) - (1 - (sigma1 - 1) / sigma1)) / 
                                   ((1 - (sigma1 - 1) / sigma1) * (PBH1 + tau1))))
    
    return E_F, derivative_E_F

def gen_2d_data(variable_parameters,parameters):
    variable_parameters["var1"]["values"] = np.linspace(variable_parameters["var1"]["min"], variable_parameters["var1"]["max"], variable_parameters["var1"]["reps"])
    variable_parameters["var2"]["values"] = np.linspace(variable_parameters["var2"]["min"], variable_parameters["var2"]["max"], variable_parameters["var2"]["reps"])

    variable_parameters["var1"]["grid"], variable_parameters["var2"]["grid"] = np.meshgrid(variable_parameters["var1"]["values"], variable_parameters["var2"]["values"], indexing='ij')

    parameters[variable_parameters["var1"]["property"]] = variable_parameters["var1"]["grid"]
    parameters[variable_parameters["var2"]["property"]] = variable_parameters["var2"]["grid"]

    return variable_parameters, parameters

def plot_contours(scen, B_values, tau_values, E_F_values, derivative_values, axis_val):
    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    fig.suptitle(scen)

    # Plot E_F
    ax1 = axs[0]
    if axis_val:
        CS1 = ax1.contourf(B_values, tau_values, E_F_values, cmap='viridis')
        #ax1.set_xlabel(r'Budget, $B$')
        ax1.set_ylabel(r'Sector 1 carbon price, $\tau_1$')
    else:
        CS1 = ax1.contourf(tau_values,B_values, E_F_values.T, cmap='viridis')
        #ax1.set_xlabel(r'Sector 1 carbon price, $\tau_1$')
        ax1.set_ylabel(r'Budget, $B$')
    cbar1 = fig.colorbar(CS1, ax=ax1)
    cbar1.set_label(r'Emissions Flow, $E_F$')

    #ax1.set_title('E_F')
    #ax1.set_xlabel(r'Budget, $B$')

    # Plot derivative of E_F with respect to tau1
    ax2 = axs[1]
    if axis_val:
        CS2 = ax2.contourf(B_values, tau_values, derivative_values, cmap='plasma')
        ax2.set_xlabel(r'Budget, $B$')
        ax2.set_ylabel(r'Sector 1 carbon price, $\tau_1$')
    else:
        CS2 = ax2.contourf(tau_values,B_values, derivative_values.T, cmap='plasma')
        ax2.set_xlabel(r'Sector 1 carbon price, $\tau_1$')
        ax2.set_ylabel(r'Budget, $B$')

    cbar2 = fig.colorbar(CS2, ax=ax2)
    cbar2.set_label(r'Derivative of Emissions Flow, $\frac{\partial E_F}{\partial \tau_m}$')

    #ax2.set_title('Derivative of E_F with respect to tau1')


    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/2_sector_E_E_derv_" + scen + "_" + str(axis_val)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_contours_2_scen(B_values, tau_values, E_F_values1, derivative_values1, E_F_values2, derivative_values2, axis_val):
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey='row')

    # Plot E_F for Scenario 1
    ax1 = axs[0][0]
    if axis_val:
        CS1 = ax1.contourf(B_values, tau_values, E_F_values1, cmap='viridis')
        ax1.set_ylabel(r'Sector 1 carbon price, $\tau_1$')
    else:
        CS1 = ax1.contourf(tau_values, B_values, E_F_values1.T, cmap='viridis')
        ax1.set_ylabel(r'Budget, $B$')


    ax1.set_title(r'Scenario 1')

    # Plot derivative of E_F with respect to tau1 for Scenario 1
    ax2 = axs[1][0]
    if axis_val:
        CS2 = ax2.contourf(B_values, tau_values, derivative_values1, cmap='plasma')
        ax2.set_xlabel(r'Budget, $B$')
        ax2.set_ylabel(r'Sector 1 carbon price, $\tau_1$')
    else:
        CS2 = ax2.contourf(tau_values, B_values, derivative_values1.T, cmap='plasma')
        ax2.set_xlabel(r'Sector 1 carbon price, $\tau_1$')
        ax2.set_ylabel(r'Budget, $B$')

    # Plot E_F for Scenario 2
    ax3 = axs[0][1]
    if axis_val:
        CS3 = ax3.contourf(B_values, tau_values, E_F_values2, cmap='viridis')
    else:
        CS3 = ax3.contourf(tau_values, B_values, E_F_values2.T, cmap='viridis')
    ax3.set_title(r'Scenario 2')
    cbar3 = fig.colorbar(CS3, ax=[ax1, ax3])
    cbar3.set_label(r'Emissions Flow, $E_F$')

    # Plot derivative of E_F with respect to tau1 for Scenario 2
    ax4 = axs[1][1]
    if axis_val:
        CS4 = ax4.contourf(B_values, tau_values, derivative_values2, cmap='plasma')
        ax4.set_xlabel(r'Budget, $B$')
    else:
        CS4 = ax4.contourf(tau_values, B_values, derivative_values2.T, cmap='plasma')
        ax4.set_xlabel(r'Sector 2 carbon price, $\tau_1$')

    # Create a single color bar for each row
    cbar4 = fig.colorbar(CS4, ax=[ax2, ax4])
    cbar4.set_label(r'Derivative of Emissions Flow, $\frac{\partial E_F}{\partial \tau_1}$')

    #plt.tight_layout()

    # Save plot
    plotName = fileName + "/Plots"
    f = plotName + "/double_2_sector_E_E_derv_" + str(axis_val)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_contours_gen( fileName,variable_parameters, E_F_values, derivative_values, levels,axis_val, scenario):
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 5),constrained_layout = True)

    var1_values = variable_parameters["var1"]["values"]
    var2_values = variable_parameters["var2"]["values"]
    var1_title = variable_parameters["var1"]["title"]
    var2_title = variable_parameters["var2"]["title"]
    #fig.suptitle("Scenario = "+ str(scenario))

    # Plot E_F
    ax1 = axs[0]
    if axis_val:
        CS1 = ax1.contourf(var1_values, var2_values, E_F_values, cmap='viridis', levels = levels)
        ax1.set_xlabel(var1_title)
        ax1.set_ylabel(var2_title)
    else:
        CS1 = ax1.contourf(var2_values,var1_values, E_F_values.T, cmap='viridis', levels = levels)
        ax1.set_xlabel(var2_title)
        ax1.set_ylabel(var1_title)
    cbar1 = fig.colorbar(CS1, ax=ax1)
    cbar1.set_label(r'Emissions Flow, $E_F$')
    # Plot derivative of E_F with respect to tau1
    ax2 = axs[1]
    if axis_val:
        CS2 = ax2.contourf(var1_values, var2_values, derivative_values, cmap='plasma', levels = levels)
        ax2.set_xlabel(var1_title)
        ax2.set_ylabel(var2_title)
    else:
        CS2 = ax2.contourf(var2_values,var1_values, derivative_values.T, cmap='plasma', levels = levels)
        ax2.set_xlabel(var2_title)
        ax2.set_ylabel(var1_title)

    cbar2 = fig.colorbar(CS2, ax=ax2)
    cbar2.set_label(r'Derivative of Emissions Flow, $\frac{\partial E_F}{\partial \tau_m}$')

    #plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/gen_2_sector_E_E_derv_" + variable_parameters["var1"]["property"] + "_" + variable_parameters["var1"]["property"] + str(axis_val)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_contours_gen_second_order( fileName,variable_parameters, E_F_tau1_tau2_values, levels,axis_val, scenario):
    # Plotting
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5),constrained_layout = True)

    var1_values = variable_parameters["var1"]["values"]
    var2_values = variable_parameters["var2"]["values"]
    var1_title = variable_parameters["var1"]["title"]
    var2_title = variable_parameters["var2"]["title"]
    #fig.suptitle("Scenario = "+ str(scenario))

    if axis_val:
        CS1 = ax1.contourf(var1_values, var2_values, E_F_tau1_tau2_values, cmap='viridis', levels = levels)
        ax1.set_xlabel(var1_title)
        ax1.set_ylabel(var2_title)
    else:
        CS1 = ax1.contourf(var2_values,var1_values, E_F_tau1_tau2_values.T, cmap='viridis', levels = levels)
        ax1.set_xlabel(var2_title)
        ax1.set_ylabel(var1_title)
    cbar1 = fig.colorbar(CS1, ax=ax1)
    cbar1.set_label(r'Second order partial derivative, $\frac{\partial E_F}{\partial \tau_2\partial \tau_1}$')
 
    plotName = fileName + "/Plots"
    f = plotName + "/gen_2_sector_second_order_" + variable_parameters["var1"]["property"] + "_" + variable_parameters["var1"]["property"] + str(axis_val)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_line_with_colorbar(fileName, variable_parameters, E_F_values, derivative_values, axis_val, scenario):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout = True)
    #fig.suptitle("Scenario = "+ str(scenario))
    var1_values = variable_parameters["var1"]["values"]
    var2_values = variable_parameters["var2"]["values"]
    var1_title = variable_parameters["var1"]["title"]
    var2_title = variable_parameters["var2"]["title"]

    if axis_val:
        x_values = var1_values
        z_values = var2_values
        xlabel = var1_title
        zlabel = var2_title
        y1_values = E_F_values
        y2_values = derivative_values
    else:
        x_values = var2_values
        z_values = var1_values
        xlabel = var2_title
        zlabel = var1_title
        y1_values = E_F_values.T
        y2_values = derivative_values.T

    y1label = r'Emissions Flow, $E_F$'
    y2label = r'Derivative of Emissions Flow, $\frac{\partial E_F}{\partial \tau_m}$'
    
    # Setting labels and legend
    axs[0].set_xlabel(xlabel)
    axs[1].set_xlabel(xlabel)
    axs[0].set_ylabel(y1label)
    axs[1].set_ylabel(y2label)

    cmap = plt.get_cmap("cividis")
    #c = Normalize(vmin=min(z_values), vmax=max(z_values))
    c = Normalize()(z_values)

    """
    I DONT UNDERSTAND WHY ITS y1_values[:,i] BUT IT WORKS
    """
    if axis_val:
        for i, z_val in enumerate(z_values):
            axs[0].plot(x_values, y1_values[:,i], color = cmap(c[i]))
            axs[1].plot(x_values, y2_values[:,i], color = cmap(c[i]))
    else:    
        for i, z_val in enumerate(z_values):
            axs[0].plot(x_values, y1_values[:,i], color = cmap(c[i]))
            axs[1].plot(x_values, y2_values[:,i], color = cmap(c[i]))

    axs[0].grid()
    axs[1].grid()

    # Adding colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( z_values), vmax=max(z_values))), ax=axs.ravel()
    )
    cbar.set_label(zlabel)

    # Saving the plot
    #check_other_folder()
    plotName = fileName + "/Plots"
    f = plotName + "/lines_2_sector_E_E_derv_" + variable_parameters["var1"]["property"] + "_" + variable_parameters["var1"]["property"] + str(axis_val)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_quad(fileName, variable_parameters, E_F_values, partial_derivative_value_tau1, partial_derivative_value_tau2, data_EF_tau1_tau2,  axis_val, scenario, levels):

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 10),constrained_layout = True)

    var1_values = variable_parameters["var1"]["values"]
    var2_values = variable_parameters["var2"]["values"]
    var1_title = variable_parameters["var1"]["title"]
    var2_title = variable_parameters["var2"]["title"]
    #fig.suptitle("Scenario = "+ str(scenario))

    # Plot E_F
    ax1 = axs[0][0]
    if axis_val:
        CS1 = ax1.contourf(var1_values, var2_values, E_F_values, cmap='viridis', levels = levels)
        ax1.set_xlabel(var1_title)
        ax1.set_ylabel(var2_title)
    else:
        CS1 = ax1.contourf(var2_values,var1_values, E_F_values.T, cmap='viridis', levels = levels)
        ax1.set_xlabel(var2_title)
        ax1.set_ylabel(var1_title)
    cbar1 = fig.colorbar(CS1, ax=ax1)
    cbar1.set_label(r'Emissions Flow, $E_F$')

    # Plot derivative of E_F with respect to tau1
    ax2 = axs[0][1]
    if axis_val:
        CS2 = ax2.contourf(var1_values, var2_values,  data_EF_tau1_tau2, cmap='cividis', levels = levels)
        ax2.set_xlabel(var1_title)
        ax2.set_ylabel(var2_title)
    else:
        CS2 = ax2.contourf(var2_values,var1_values,  data_EF_tau1_tau2.T, cmap='cividis', levels = levels)
        ax2.set_xlabel(var2_title)
        ax2.set_ylabel(var1_title)

    cbar2 = fig.colorbar(CS2, ax=ax2)
    cbar2.set_label(r'Second order partial derivative, $\frac{\partial E_F}{\partial \tau_2\partial \tau_1}$')

    # Plot derivative of E_F with respect to tau1
    ax3 = axs[1][0]
    if axis_val:
        CS3 = ax3.contourf(var1_values, var2_values, partial_derivative_value_tau1, cmap='plasma', levels = levels)
        ax3.set_xlabel(var1_title)
        ax3.set_ylabel(var2_title)
    else:
        CS3 = ax3.contourf(var2_values,var1_values, partial_derivative_value_tau1.T, cmap='plasma', levels = levels)
        ax3.set_xlabel(var2_title)
        ax3.set_ylabel(var1_title)

    cbar3 = fig.colorbar(CS3, ax=ax3)
    cbar3.set_label(r'Derivative of Emissions Flow, $\frac{\partial E_F}{\partial \tau_1}$')

    # Plot derivative of E_F with respect to tau1
    ax4 = axs[1][1]
    if axis_val:
        CS4 = ax4.contourf(var1_values, var2_values,  partial_derivative_value_tau2, cmap='plasma', levels = levels)
        ax4.set_xlabel(var1_title)
        ax4.set_ylabel(var2_title)
    else:
        CS4 = ax4.contourf(var2_values,var1_values,  partial_derivative_value_tau2.T, cmap='plasma', levels = levels)
        ax4.set_xlabel(var2_title)
        ax4.set_ylabel(var1_title)

    cbar4 = fig.colorbar(CS4, ax=ax4)
    cbar4.set_label(r'Derivative of Emissions Flow, $\frac{\partial E_F}{\partial \tau_2}$')

    #plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/gen_2_sector_E_E_derv_" + variable_parameters["var1"]["property"] + "_" + variable_parameters["var1"]["property"] + str(axis_val)
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def run_plots(root,LOAD, init_params, scenario, LOAD_filename = "filename"):
    
    if LOAD:
        #fileName, variable_parameters, parameters_run, scenario, init_params = set_up_data(root = "2_sector_analytic",LOAD = 0, init_params = init_params, scenario = scenario)
        print("LOADED D")
        fileName = LOAD_filename
        print("fileName", fileName)
        parameters_run = load_object(fileName + "/Data","parameters_run")
        variable_parameters = load_object(fileName + "/Data","variable_parameters")
        data_E_F_1 = load_object(fileName + "/Data","data_E_F")
        data_derv_E_F_1 = load_object(fileName + "/Data","data_derv_E_F")
        data_EF_tau1_tau2 = load_object(fileName + "/Data","data_EF_tau1_tau2")
        scenario = load_object(fileName + "/Data","scenario")
    else:
        fileName, variable_parameters, parameters_run, scenario, init_params = set_up_data(root, init_params, scenario)
        
        #data_E_F_1, data_derv_E_F_1 = calculate_data_alt(parameters_run)
        data_E_F_1, data_derv_E_F_1 = calc_emissions_and_derivative(parameters_run)

        createFolder(fileName)

        save_object(variable_parameters, fileName + "/Data", "variable_parameters")
        save_object(parameters_run, fileName + "/Data", "parameters_run")
        save_object(data_derv_E_F_1, fileName + "/Data", "data_derv_E_F")
        save_object(data_E_F_1, fileName + "/Data", "data_E_F")
        save_object(scenario, fileName + "/Data", "scenario")
        save_object(init_params, fileName + "/Data", "init_params")
    
    levels = 20
    if scenario not in (2,3):
        plot_contours_gen(fileName,variable_parameters, data_E_F_1, data_derv_E_F_1,levels, 1, scenario)
    plot_line_with_colorbar(fileName,variable_parameters, data_E_F_1, data_derv_E_F_1, 0, scenario)
    plot_line_with_colorbar(fileName, variable_parameters, data_E_F_1, data_derv_E_F_1, 1, scenario)

    #plot_contours(fileName, "Scenario 1",B_values, tau_values, data_E_F_1, data_derv_E_F_1, axis_val)
    #plot_contours(fileName, "Scenario 2",B_values, tau_values, data_E_F_2, data_derv_E_F_2, axis_val)
    #plot_contours_2_scen(fileName, B_values, tau_values, data_E_F_1, data_derv_E_F_1, data_E_F_2, data_derv_E_F_2, axis_val)

    plt.show()

def analytic_derivatives():

    # Define symbols
    tau1_sym, tau2_sym, B_sym, nu_sym, h1_sym, h2_sym, PBH1_sym, PBH2_sym, A1_sym, A2_sym, PL1_sym, PL2_sym, sigma1_sym, sigma2_sym, a1_sym, a2_sym = symbols('tau1 tau2 B nu h1 h2 PBH1 PBH2 A1 A2 PL1 PL2 sigma1 sigma2 a1 a2')
    
    # Define BD equation
    BD_sym = B_sym - h1_sym * (PBH1_sym + tau1_sym) - h2_sym * (PBH2_sym + tau2_sym)
    #derv_BD_sym = simplify(diff(BD_sym, tau1_sym))
    #manual_derv_BD_sym = simplify(-h1_sym)

    # Define Omega1 equation
    Omega1_sym = ((PBH1_sym + tau1_sym) * A1_sym / (PL1_sym * (1 - A1_sym)))**sigma1_sym

    # Define Omega2 equation
    Omega2_sym = ((PBH2_sym + tau2_sym) * A2_sym / (PL2_sym * (1 - A2_sym)))**sigma2_sym

    # Define chi1 equation
    chi1_sym = (a1_sym / (PBH1_sym + tau1_sym)) * ((A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) + (1 - A1_sym))**(((nu_sym - 1) * sigma1_sym) / (nu_sym * (sigma1_sym - 1)))
    #derv_chi1_sym = diff(chi1_sym, tau1_sym)
    #manual_derv_chi1_sym = (chi1_sym / (PBH1_sym + tau1_sym)) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) / (nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1)

    # Define chi2 equation
    chi2_sym = (a2_sym / (PBH2_sym + tau2_sym)) * ((A2_sym * Omega2_sym**((sigma2_sym - 1) / sigma2_sym)) + (1 - A2_sym))**(((nu_sym - 1) * sigma2_sym) / (nu_sym * (sigma2_sym - 1)))
    #derv_chi2_sym = diff(chi2_sym, tau1_sym)
    #manual_derv_chi2_sym = 0

#############################################################################################
    # Define Z equation
    Z_sym = (chi1_sym**nu_sym * (Omega1_sym * PL1_sym + PBH1_sym + tau1_sym)) + (chi2_sym**nu_sym * (Omega2_sym * PL2_sym + PBH2_sym + tau2_sym))
    #derv_Z_sym = diff(Z_sym, tau1_sym)
    #manual_derv_Z_sym = ((Omega1_sym * PL1_sym + PBH1_sym + tau1_sym) * nu_sym * chi1_sym**nu_sym) / (PBH1_sym + tau1_sym) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym))/(nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1) + chi1_sym**nu_sym * (PL1_sym * (sigma1_sym * Omega1_sym) / (PBH1_sym + tau1_sym) + 1)

#########################################################################################################
    #H1
    #DERIVE IT SUING SYMPY
    #H_1 = (BD_sym*chi1_sym**nu_sym/Z_sym) + h1_sym
    #partial_H1_tau1_sympy = diff(H_1, tau1_sym)

#########################################################################################################
    #H2
    #DERIVE IT SUING SYMPY
    #H_2 = (BD_sym*chi2_sym**nu_sym/Z_sym) + h2_sym
    #partial_H2_tau1_sympy = diff(H_2, tau1_sym)

##########################################################################################################

    #quit()
    # Define EF equation with BD substituted
    EF = (BD_sym * (chi1_sym**nu_sym + chi2_sym**nu_sym) / Z_sym) + h1_sym + h2_sym

    # Differentiate EF with respect to tau1
    partial_derivative_EF_tau1 = diff(EF, tau1_sym)
    partial_derivative_EF_tau2 = diff(EF, tau2_sym)

    #print("EF and first derv done")
    EF_tau1_tau2  = diff(partial_derivative_EF_tau1, tau2_sym)
    #print("SECOND ORDER DERIVATIVE DONE")
    #manual_derv_EF = BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * ((chi1_sym / (PBH1_sym + tau1_sym)) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) / (nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1)) - Z_sym**(-1) * h1_sym * (chi1_sym**nu_sym + chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (chi1_sym**nu_sym + chi2_sym**nu_sym) * ((Omega1_sym * PL1_sym + PBH1_sym + tau1_sym) * nu_sym * chi1_sym**nu_sym / (PBH1_sym + tau1_sym) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) / (nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1) + chi1_sym**nu_sym * (PL1_sym * sigma1_sym * Omega1_sym / (PBH1_sym + tau1_sym) + 1))

    return EF, partial_derivative_EF_tau1, partial_derivative_EF_tau2, EF_tau1_tau2 

def analytic_derivatives_alt():
    subs_dict = {#BASE PARAMS
                "A1": 0.5,
                "A2": 0.5,
                "tau1": 1.2,#1,
                "tau2": 0.4,
                "a1": 0.5,
                "a2": 0.5,
                "sigma1": 2,
                "sigma2": 2,
                "nu": 2,
                "PL1": 1,
                "PL2": 1,
                "PBH1": 1,
                "PBH2": 1,
                "h1": 0.7,
                "h2": 0.3,
                "B": 5,
                "gamma1": 0.9,#0.8,
                "gamma2": 2.5,#1.1
    }

    """
    subs_dict = {#BASE PARAMS
                "A1": 0.5,
                "A2": 0.5,
                "tau1": 1,#1,
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
                "B": 5,
                "gamma1": 1,#0.8,
                "gamma2": 1,#1.1
    }
    """
        
    # Define symbols
    tau1_sym, tau2_sym, B_sym, nu_sym, h1_sym, h2_sym, PBH1_sym, PBH2_sym, A1_sym, A2_sym, PL1_sym, PL2_sym, sigma1_sym, sigma2_sym, a1_sym, a2_sym, gamma1_sym, gamma2_sym = symbols('tau1 tau2 B nu h1 h2 PBH1 PBH2 A1 A2 PL1 PL2 sigma1 sigma2 a1 a2 gamma1 gamma2')
    
    # Define BD equation
    BD_sym = B_sym - h1_sym * (PBH1_sym + gamma1_sym*tau1_sym) - h2_sym * (PBH2_sym + gamma2_sym*tau2_sym)
    #derv_BD_sym = diff(BD_sym, tau1_sym)
    #manual_derv_BD_sym = -gamma1_sym*h1_sym

    #BD_sym_alt = B_sym - h1_sym * (PBH1_sym + tau1_sym) - h2_sym * (PBH2_sym + tau2_sym)
    #derv_BD_sym_alt = diff(BD_sym_alt, tau1_sym)
    #manual_derv_BD_sym_alt = -h1_sym
    #print("Vlaues of BD", BD_sym.subs(subs_dict))#,BD_sym_alt.subs(subs_dict)
    #print("Vlaues of derv BD", derv_BD_sym.subs(subs_dict),manual_derv_BD_sym.subs(subs_dict))#,derv_BD_sym_alt.subs(subs_dict),manual_derv_BD_sym_alt.subs(subs_dict)

    # Define Omega1 equation
    Omega1_sym = ((PBH1_sym + gamma1_sym*tau1_sym) * A1_sym / (PL1_sym * (1 - A1_sym)))**sigma1_sym
    #derv_Omega1_sym = diff(Omega1_sym, tau1_sym)
    #manual_derv_Omega1_sym = gamma1_sym*sigma1_sym*Omega1_sym/(PBH1_sym + gamma1_sym*tau1_sym)

    #Omega1_sym_alt = ((PBH1_sym + tau1_sym) * A1_sym / (PL1_sym * (1 - A1_sym)))**sigma1_sym
    #derv_Omega1_sym_alt = diff(Omega1_sym_alt, tau1_sym)
    #manual_derv_Omega1_sym_alt = sigma1_sym*Omega1_sym_alt/(PBH1_sym + tau1_sym)

    #print("Vlaues of omega1", Omega1_sym.subs(subs_dict))#,Omega1_sym_alt.subs(subs_dict)
    #print("Vlaues of derv omega1 ",  derv_Omega1_sym.subs(subs_dict), manual_derv_Omega1_sym.subs(subs_dict) )#, derv_Omega1_sym_alt.subs(subs_dict),manual_derv_Omega1_sym_alt.subs(subs_dict) 

    # Define Omega2 equation
    Omega2_sym = ((PBH2_sym + tau2_sym) * A2_sym / (PL2_sym * (1 - A2_sym)))**sigma2_sym
    #derv_Omega2_sym = diff(Omega2_sym, tau1_sym)
    #manual_derv_Omega2_sym = 0

    #Omega2_sym_alt = ((PBH2_sym + gamma2_sym*tau2_sym) * A2_sym / (PL2_sym * (1 - A2_sym)))**sigma2_sym
    #derv_Omega2_sym_alt = diff(Omega2_sym_alt, tau1_sym)
    #manual_derv_Omega2_sym_alt = 0

    #print("Vlaues of omega2", Omega2_sym.subs(subs_dict))#,Omega2_sym_alt.subs(subs_dict)
    #print("Vlaues of derv omega2 ",  derv_Omega2_sym.subs(subs_dict), manual_derv_Omega2_sym)#, derv_Omega2_sym_alt.subs(subs_dict) ,manual_derv_Omega2_sym_alt
    

    # Define chi1 equation
    chi1_sym = (a1_sym / (PBH1_sym + gamma1_sym*tau1_sym)) * ((A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) + (1 - A1_sym))**(((nu_sym - 1) * sigma1_sym) / (nu_sym * (sigma1_sym - 1)))
    #derv_chi1_sym = diff(chi1_sym,tau1_sym)
    #manual_derv_chi1_sym_73 = -((gamma1_sym*a1_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))**(((nu_sym - 1) * sigma1_sym) / (nu_sym * (sigma1_sym - 1)))) / ((PBH1_sym + gamma1_sym * tau1_sym)**2)) + ((a1_sym * A1_sym * (nu_sym - 1) * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))**((nu_sym - sigma1_sym) / (nu_sym * (sigma1_sym - 1)))) / (nu_sym * (PBH1_sym + gamma1_sym * tau1_sym) * Omega1_sym**(1 / sigma1_sym))) * diff(Omega1_sym, tau1_sym)
    #below lines should be identical
    #manual_derv_chi1_sym_73_subbed = -((gamma1_sym*a1_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))**(((nu_sym - 1) * sigma1_sym) / (nu_sym * (sigma1_sym - 1)))) / ((PBH1_sym + gamma1_sym * tau1_sym)**2)) + ((a1_sym * A1_sym * (nu_sym - 1) * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))**((nu_sym - sigma1_sym) / (nu_sym * (sigma1_sym - 1)))) / (nu_sym * (PBH1_sym + gamma1_sym * tau1_sym) * Omega1_sym**(1 / sigma1_sym))) * (manual_derv_Omega1_sym)
    
    #manual_derv_chi1_sym = (gamma1_sym*chi1_sym / (PBH1_sym + gamma1_sym*tau1_sym)) * (((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) / (nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym)))) - 1)
    
    #chi1_sym_alt = (a1_sym / (PBH1_sym + tau1_sym)) * ((A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) + (1 - A1_sym))**(((nu_sym - 1) * sigma1_sym) / (nu_sym * (sigma1_sym - 1)))
    #derv_chi1_sym_alt = diff(chi1_sym_alt, tau1_sym)
    #manual_derv_chi1_sym_alt = (chi1_sym_alt / (PBH1_sym + tau1_sym)) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym_alt**((sigma1_sym - 1) / sigma1_sym)) / (nu_sym * (A1_sym * Omega1_sym_alt**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1)
    #print("Vlaues of chi1", chi1_sym.subs(subs_dict))#,chi1_sym_alt.subs(subs_dict)
    #print("Vlaues of derv chi1 ",  derv_chi1_sym.subs(subs_dict), manual_derv_chi1_sym_73.subs(subs_dict), manual_derv_chi1_sym_73_subbed.subs(subs_dict), manual_derv_chi1_sym.subs(subs_dict))#, derv_chi1_sym_alt.subs(subs_dict)), manual_derv_chi1_sym_alt.subs(subs_dict)
    #quit()
    # Define chi2 equation
    chi2_sym = (a2_sym / (PBH2_sym + gamma2_sym*tau2_sym)) * ((A2_sym * Omega2_sym**((sigma2_sym - 1) / sigma2_sym)) + (1 - A2_sym))**(((nu_sym - 1) * sigma2_sym) / (nu_sym * (sigma2_sym - 1)))
    #derv_chi2_sym = diff(chi2_sym, tau1_sym)
    #manual_derv_chi2_sym = 0

    #chi2_sym_alt = (a2_sym / (PBH2_sym + tau2_sym)) * ((A2_sym * Omega2_sym_alt**((sigma2_sym - 1) / sigma2_sym)) + (1 - A2_sym))**(((nu_sym - 1) * sigma2_sym) / (nu_sym * (sigma2_sym - 1)))
    #derv_chi2_sym_alt = diff(chi2_sym_alt, tau1_sym)
    #manual_derv_chi2_sym_alt = 0

    #print("Vlaues of chi2", chi2_sym.subs(subs_dict))#,chi2_sym_alt.subs(subs_dict)
    #print("Vlaues of derv chi2 ",  derv_chi2_sym.subs(subs_dict), derv_chi2_sym_alt.subs(subs_dict))

#############################################################################################
    # Define Z equation
    Z_sym = (chi1_sym**nu_sym * (Omega1_sym * PL1_sym + PBH1_sym + gamma1_sym*tau1_sym)) + (chi2_sym**nu_sym * (Omega2_sym * PL2_sym + PBH2_sym + gamma2_sym*tau2_sym))
    #derv_Z_sym = diff(Z_sym, tau1_sym)
    #manual_derv_Z_sym = ((Omega1_sym * PL1_sym + PBH1_sym + gamma1_sym*tau1_sym) * gamma1_sym * nu_sym * chi1_sym**nu_sym) / (PBH1_sym + gamma1_sym*tau1_sym) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym))/(nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1) + gamma1_sym*chi1_sym**nu_sym * (PL1_sym * (sigma1_sym * Omega1_sym) / (PBH1_sym + gamma1_sym*tau1_sym) + 1)
    
    #Z_sym_alt = (chi1_sym_alt**nu_sym * (Omega1_sym_alt * PL1_sym + PBH1_sym + tau1_sym)) + (chi2_sym_alt**nu_sym * (Omega2_sym_alt * PL2_sym + PBH2_sym + tau2_sym))
    #derv_Z_sym_alt = diff(Z_sym_alt, tau1_sym)
    #manual_derv_Z_sym_alt = ((Omega1_sym_alt * PL1_sym + PBH1_sym + tau1_sym) * nu_sym * chi1_sym_alt**nu_sym) / (PBH1_sym + tau1_sym) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym_alt**((sigma1_sym - 1) / sigma1_sym))/(nu_sym * (A1_sym * Omega1_sym_alt**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1) + chi1_sym_alt**nu_sym * (PL1_sym * (sigma1_sym * Omega1_sym_alt) / (PBH1_sym + tau1_sym) + 1)
    #print("Vlaues of Z", Z_sym.subs(subs_dict))#,Z_sym_alt.subs(subs_dict)
    #print("Vlaues of derv Z", derv_Z_sym.subs(subs_dict), manual_derv_Z_sym.subs(subs_dict))#, derv_Z_sym_alt.subs(subs_dict), manual_derv_Z_sym_alt.subs(subs_dict)
    #quit()

##########################################################################################################

    # Define EF equation with BD substituted
    EF = (BD_sym * (gamma1_sym*chi1_sym**nu_sym + gamma2_sym*chi2_sym**nu_sym) / Z_sym) + gamma1_sym*h1_sym + gamma2_sym*h2_sym
    #derv_EF = diff(EF, tau1_sym)
    #manual_derv_EF = 
    #manual_derv_87  = gamma1_sym * BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * diff(chi1_sym, tau1_sym) - Z_sym**(-1) * gamma1_sym * h1_sym * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) * diff(Z_sym, tau1_sym)
    #manual_derv_87_subbed  = gamma1_sym * BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * manual_derv_chi1_sym - Z_sym**(-1) * gamma1_sym * h1_sym * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) * manual_derv_Z_sym
    
    manual_derv_EF_tau1 = gamma1_sym * BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * ((gamma1_sym*chi1_sym / (PBH1_sym + gamma1_sym*tau1_sym)) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) / (nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1)) - Z_sym**(-1) * gamma1_sym * h1_sym * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) * (((Omega1_sym * PL1_sym + PBH1_sym + gamma1_sym*tau1_sym) * gamma1_sym * nu_sym * chi1_sym**nu_sym) / (PBH1_sym + gamma1_sym*tau1_sym) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym))/(nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1) + gamma1_sym*chi1_sym**nu_sym * (PL1_sym * (sigma1_sym * Omega1_sym) / (PBH1_sym + gamma1_sym*tau1_sym) + 1))

    #EF_alt = (BD_sym_alt * (chi1_sym_alt**nu_sym + chi2_sym_alt**nu_sym) / Z_sym_alt) + h1_sym + h2_sym
    #derv_EF_alt = diff(EF_alt, tau1_sym)
    #manual_derv_EF_alt = BD_sym_alt * Z_sym_alt**(-1) * nu_sym * chi1_sym_alt**(nu_sym - 1) * ((chi1_sym_alt / (PBH1_sym + tau1_sym)) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym_alt**((sigma1_sym - 1) / sigma1_sym)) / (nu_sym * (A1_sym * Omega1_sym_alt**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1)) - Z_sym_alt**(-1) * h1_sym * (chi1_sym_alt**nu_sym + chi2_sym_alt**nu_sym) - BD_sym_alt * Z_sym_alt**(-2) * (chi1_sym_alt**nu_sym + chi2_sym_alt**nu_sym) * ((Omega1_sym_alt * PL1_sym + PBH1_sym + tau1_sym) * nu_sym * chi1_sym_alt**nu_sym / (PBH1_sym + tau1_sym) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym_alt**((sigma1_sym - 1) / sigma1_sym)) / (nu_sym * (A1_sym * Omega1_sym_alt**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1) + chi1_sym_alt**nu_sym * (PL1_sym * sigma1_sym * Omega1_sym_alt / (PBH1_sym + tau1_sym) + 1))
    #print("Vlaues of EF", EF.subs(subs_dict))#,EF_alt.subs(subs_dict)
    #print("Vlaues of derv E", derv_EF.subs(subs_dict) , manual_derv_87.subs(subs_dict), manual_derv_87_subbed.subs(subs_dict), manual_derv_EF.subs(subs_dict))#, derv_EF_alt.subs(subs_dict), manual_derv_EF.subs(subs_dict)
    #quit()
    #TEST 107
    #manual_derv_87  = gamma1_sym * BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * diff(chi1_sym, tau1_sym) - Z_sym**(-1) * gamma1_sym * h1_sym * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) * diff(Z_sym, tau1_sym)

    #manual_derv_88_alt = gamma1_sym * BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * manual_derv_chi1_sym - Z_sym**(-1) * gamma1_sym * h1_sym * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) * manual_derv_Z_sym

    #manual_derv_EF_sym_107 = BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * manual_derv_chi1_sym - Z_sym**(-1) * h1_sym * (chi1_sym**nu_sym + chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (chi1_sym**nu_sym + chi2_sym**nu_sym) * manual_derv_Z_sym

    # Combine terms

    #derv_EF_sym_value = partial_derivative_EF_tau1.subs(subs_dict)
    #manual_derv_87_value = manual_derv_87.subs(subs_dict)
    #manual_derv_88_alt_value = manual_derv_88_alt.subs(subs_dict)
    #manual_derv_EF_sym_107_value = manual_derv_EF_sym_107.subs(subs_dict)

    #print("Vlaues of derv E", derv_EF_sym_value,manual_derv_87_value,manual_derv_88_alt_value,manual_derv_EF_sym_107_value)
    #print("Manually derived expression:", manual_derivative_sym)

    #quit()
    """
    simplified_manual = simplify(manual_derivative_sym)
    print("DONE simplified_manual")
    simplified_sympy = simplify(partial_derivative_EF_tau1)
    print("DONE simplified_sympy")

    if simplified_manual == simplified_sympy:
        print("Expressions match after simplification.")
    else:
        print("Expressions do not match after simplification.")

    print("partial_derivative_EF_tau1",partial_derivative_EF_tau1)
    quit()
    """

    return EF, manual_derv_EF_tau1

def calc_emissions_and_derivative(parameters):
    
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

    # Define BD equation
    BD = B - h1 * (PBH1 + tau1) - h2 * (PBH2 + tau2)
    manual_derv_BD = -h1

    # Define Omega1 equation
    Omega1 = ((PBH1 + tau1) * A1 / (PL1 * (1 - A1)))**sigma1

    # Define Omega2 equation
    Omega2 = ((PBH2 + tau2) * A2 / (PL2 * (1 - A2)))**sigma2

    # Define chi1 equation
    chi1 = (a1 / (PBH1 + tau1)) * ((A1 * Omega1**((sigma1 - 1) / sigma1)) + (1 - A1))**(((nu - 1) * sigma1) / (nu * (sigma1 - 1)))
    manual_derv_chi1 = (chi1 / (PBH1 + tau1)) * ((sigma1 * (nu - 1) * A1 * Omega1**((sigma1 - 1) / sigma1)) / (nu * (A1 * Omega1**((sigma1 - 1) / sigma1) + (1 - A1))) - 1)

    # Define chi2 equation
    chi2 = (a2 / (PBH2 + tau2)) * ((A2 * Omega2**((sigma2 - 1) / sigma2)) + (1 - A2))**(((nu - 1) * sigma2) / (nu * (sigma2 - 1)))

    #############################################################################################
    # Define Z equation
    Z = (chi1**nu * (Omega1 * PL1 + PBH1 + tau1)) + (chi2**nu * (Omega2 * PL2 + PBH2 + tau2))
    manual_derv_Z = ((Omega1 * PL1 + PBH1 + tau1) * nu * chi1**nu) / (PBH1 + tau1) * ((sigma1 * (nu - 1) * A1 * Omega1**((sigma1 - 1) / sigma1))/(nu * (A1 * Omega1**((sigma1 - 1) / sigma1) + (1 - A1))) - 1) + chi1**nu * (PL1 * (sigma1 * Omega1) / (PBH1 + tau1) + 1)

    #quit()
    # Define EF equation with BD substituted
    EF = (BD * (chi1**nu + chi2**nu) / Z) + h1 + h2

    manual_derv_EF = BD * Z**(-1) * nu * chi1**(nu - 1) * ((chi1 / (PBH1 + tau1)) * ((sigma1 * (nu - 1) * A1 * Omega1**((sigma1 - 1) / sigma1)) / (nu * (A1 * Omega1**((sigma1 - 1) / sigma1) + (1 - A1))) - 1)) - Z**(-1) * h1 * (chi1**nu + chi2**nu) - BD * Z**(-2) * (chi1**nu + chi2**nu) * ((Omega1 * PL1 + PBH1 + tau1) * nu * chi1**nu / (PBH1 + tau1) * ((sigma1 * (nu - 1) * A1 * Omega1**((sigma1 - 1) / sigma1)) / (nu * (A1 * Omega1**((sigma1 - 1) / sigma1) + (1 - A1))) - 1) + chi1**nu * (PL1 * sigma1 * Omega1 / (PBH1 + tau1) + 1))

    return EF, manual_derv_EF

def plots_analytic(root = "2_sector_analytic",LOAD = 0, init_params = 4, scenario = 1, LOAD_filename = "filename"):
    
    if LOAD:
        #fileName, variable_parameters, parameters_run, scenario, init_params = set_up_data(root = "2_sector_analytic",LOAD = 0, init_params = init_params, scenario = scenario)
        print("LOADED DONE")
        fileName = LOAD_filename
        #print("fileName", fileName)
        parameters_run = load_object(fileName + "/Data","parameters_run")
        variable_parameters = load_object(fileName + "/Data","variable_parameters")
        data_E_F_value = load_object(fileName + "/Data","data_E_F")
        partial_derivative_tau1_value = load_object(fileName + "/Data","data_derv_E_F_tau1")
        partial_derivative_tau2_value = load_object(fileName + "/Data","data_derv_E_F_tau2")
        EF_tau1_tau2_value = load_object(fileName + "/Data","data_EF_tau1_tau2")
        scenario = load_object(fileName + "/Data","scenario")
    else:
        fileName, variable_parameters, parameters_run, scenario, init_params = set_up_data(root = root, init_params = init_params, scenario = scenario)

        EF, partial_derivative_EF_tau1, partial_derivative_EF_tau2, EF_tau1_tau2 = analytic_derivatives()

        # Determine which parameters are varied
        varied_params = [str(sym) for sym in EF.free_symbols]

        #create lambdify functions
        EF_func = lambdify(varied_params,EF, 'numpy')
        partial_derivative_tau1_func = lambdify(varied_params,partial_derivative_EF_tau1, 'numpy')
        partial_derivative_tau2_func = lambdify(varied_params,partial_derivative_EF_tau2, 'numpy')
        EF_tau1_tau2_func = lambdify(varied_params,EF_tau1_tau2, 'numpy')
        #Calc values
        data_E_F_value = EF_func(**parameters_run)
        partial_derivative_tau1_value = partial_derivative_tau1_func(**parameters_run)
        partial_derivative_tau2_value = partial_derivative_tau2_func(**parameters_run)
        EF_tau1_tau2_value = EF_tau1_tau2_func(**parameters_run)

        createFolder(fileName)

        save_object(variable_parameters, fileName + "/Data", "variable_parameters")
        save_object(parameters_run, fileName + "/Data", "parameters_run")
        save_object(data_E_F_value, fileName + "/Data", "data_E_F")
        save_object(partial_derivative_tau1_value, fileName + "/Data", "data_derv_E_F_tau1")
        save_object(partial_derivative_tau2_value, fileName + "/Data", "data_derv_E_F_tau2")
        save_object(EF_tau1_tau2_value, fileName + "/Data", "data_EF_tau1_tau2")
        save_object(scenario, fileName + "/Data", "scenario")
        save_object(init_params, fileName + "/Data", "init_params")

    #axis_val = 0
    levels = 20
    #if scenario not in (2,3):
    #    plot_contours_gen(fileName,variable_parameters, EF_value, partial_derivative_value,levels, 1, scenario)
    plot_line_with_colorbar(fileName,variable_parameters, data_E_F_value,  partial_derivative_tau1_value, 0, scenario)
    plot_line_with_colorbar(fileName, variable_parameters, data_E_F_value,  partial_derivative_tau1_value, 1, scenario)
    #plot_contours_gen_second_order(fileName,variable_parameters, data_EF_tau1_tau2,levels, 1, scenario)

    plot_quad(fileName, variable_parameters, data_E_F_value, partial_derivative_tau1_value, partial_derivative_tau2_value, EF_tau1_tau2_value,  1, scenario, levels)
    plt.show()

def calc_emissions_derv(parameters):

    EF, partial_derivative_EF_tau1 = analytic_derivatives()

    # Substitute the values into the derivative expression
    subs_dict = {sym: parameters[str(sym)] for sym in EF.free_symbols}

    EF_value = EF.subs(subs_dict)
    print("EF_value",EF_value)
    partial_derivative_value = partial_derivative_EF_tau1.subs(subs_dict)
    print("partial_derivative_value",partial_derivative_value)

    return EF_value, partial_derivative_value

def set_up_data(root = "2_sector_model", init_params = 4, scenario = 1):
    """
    Scenario 1: is that sector 1 is a basic good with high subsititutability(FOOD), similar prices between both goods but a minimum of the high carbon required. But there is low preference for this sector
    Sector 2 has low substitutability, but more attractive (LONG DISTANCE TRAVEL), also the High carbon base price is much lower, howeer there is no minimum required
    - Assume individuals are indifferent to the environment
    """
    #############################################
    match init_params:
        case 1:
            parameters_1 = {#BASE PARAMS
                "A1": 0.5,
                "A2": 0.5,
                "tau1": 0,
                "tau2": 0,
                "a1": 0.5,
                "a2": 0.5,
                "sigma1": 5,
                "sigma2": 1.5,
                "nu": 1.5,
                "PL1": 1,
                "PL2": 1,
                "PBH1": 1,
                "PBH2": 1,
                "h1": 1,
                "h2": 0,
                "B": 5
            }
        case 2:
            """
            PARAMETERISED FOR:
             - SECTOR 1 FOOD/short distance travel? - similar prices with low and high carbon options, minimum high carbon good required, not favoured, cheap!
             - Sector 2 Long distance travel -  very large difference in prices, no minimum, favoured, expensive
            """
            parameters_1 = {
                "A1": 0.5,
                "A2": 0.1,
                "tau1": 0,
                "tau2": 0,
                "a1": 0.2,
                "a2": 0.8,
                "sigma1": 3,
                "sigma2": 3,
                "nu": 3,
                "PL1": 1,
                "PL2": 10,
                "PBH1": 1,
                "PBH2": 10,
                "h1": 3,
                "h2": 0,
                "B": 10
            }
        case 3:
            parameters_1 = {
                "A1": 0.5,
                "A2": 0.5,
                "tau1": 0,
                "tau2": 0,
                "a1": 0.5,
                "a2": 0.5,
                "sigma1": 5,
                "sigma2": 1.5,
                "nu": 1.5,
                "PL1": 10,
                "PL2": 10,
                "PBH1": 10,
                "PBH2": 1,
                "h1": 0,
                "h2": 0,
                "B": 5
            }
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
                "B": 1
            }
        case 5:
            parameters_1 = {
                "A1": 0.5,
                "A2": 0.5,
                "tau1": 0,
                "tau2": 0,
                "a1": 0.5,
                "a2": 0.5,
                "sigma1": 5,
                "sigma2": 5,
                "nu": 5,
                "PL1": 2,
                "PL2": 2,
                "PBH1": 2,
                "PBH2": 1,
                "h1": 0,
                "h2": 0,
                "B": 10
            }
    ##################################################
    match scenario:
        case 1:
            variable_parameters = {
                "var1":{
                    "property": "tau1",
                    "min":0,
                    "max": 1,
                    "reps": 1000,
                    "title": r'Sector 1 carbon price, $\tau_1$'
                },
                "var2":{
                    "property": "tau2",
                    "min":0,
                    "max": 1,
                    "reps": 1000,
                    "title": r'Sector 2 carbon price, $\tau_2$'
                },
            }
        case 2:
            variable_parameters = {
                "var1":{
                    "property": "tau1",
                    "min":0,
                    "max": 1,
                    "reps": 1000,
                    "title": r'Sector 1 carbon price, $\tau_1$'
                },
                "var2":{
                    "property": "B",
                    "min":3,
                    "max": 10,
                    "reps": 10,
                    "title": r'Budget, $B$'
                },
            }
        case 3:
            variable_parameters = {
                "var1":{
                    "property": "tau2",
                    "min":0,
                    "max": 1,
                    "reps": 1000,
                    "title": r'Sector 2 carbon price, $\tau_2$'
                },
                "var2":{
                    "property": "B",
                    "min":3,
                    "max": 10,
                    "reps": 10,
                    "title": r'Budget, $B$'
                },
            }
        case 4:
            variable_parameters = {
                "var1":{
                    "property": "tau1",
                    "min":0,
                    "max": 1,
                    "reps": 1000,
                    "title": r'Sector 1 carbon price, $\tau_1$'
                },
                "var2":{
                    "property": "PBH2",
                    "min":0.5,
                    "max": 1.5,
                    "reps": 1000,
                    "title": r'Sector 2 high-carbon base price, $P_{B,H,2}$'
                }
            }
        case 5:
            variable_parameters = {
                "var1":{
                    "property": "tau1",
                    "min":0,
                    "max": 1,
                    "reps": 1000,
                    "title": r'Sector 1 carbon price, $\tau_1$'
                },
                "var2":{
                    "property": "PL2",
                    "min":1,
                    "max": 2,
                    "reps": 1000,
                    "title": r'Sector 2 low-carbon base price, $P_{L,2}$'
                }
            }
        case 6:
            variable_parameters = {
                "var1":{
                    "property": "tau1",
                    "min":0,
                    "max": 1,
                    "reps": 1000,
                    "title": r'Sector 1 carbon price, $\tau_1$'
                },
                "var2":{
                    "property": "h1",
                    "min":0,
                    "max": 2,
                    "reps": 1000,
                    "title": r'Sector 1 high-carbon miniumum, $h_1$'
                }
            }
        case 7:
            variable_parameters = {
                "var1":{
                    "property": "tau1",
                    "min":0,
                    "max": 1,
                    "reps": 1000,
                    "title": r'Sector 1 carbon price, $\tau_1$'
                },
                "var2":{
                    "property": "tau2",
                    "min":0,
                    "max": 10,
                    "reps": 1000,
                    "title": r'Sector 2 carbon price, $\tau_2$'
                },
            }

    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    variable_parameters, parameters_run = gen_2d_data(variable_parameters, parameters_1)

    return fileName, variable_parameters, parameters_run, scenario, init_params

def main( type_run):

    if type_run == "plots":
        #run_plots(root = "2_sector_model", LOAD = 0, init_params = 4, scenario = 1)
        run_plots(LOAD = 1, LOAD_filename = "results/2_sector_analytic_19_37_47__20_03_2024")
    elif type_run == "analytic":
        analytic_derivatives_alt()
        #analytic_derivatives()
    elif type_run == "calc_analytic":
        parameters = {#BASE PARAMS
                    "A1": 0.5,
                    "A2": 0.5,
                    "tau1": 0,
                    "tau2": 0,
                    "a1": 0.5,
                    "a2": 0.5,
                    "sigma1": 5,
                    "sigma2": 1.5,
                    "nu": 1.5,
                    "PL1": 1,
                    "PL2": 1,
                    "PBH1": 1,
                    "PBH2": 1,
                    "h1": 1,
                    "h2": 0,
                    "B": 5
        }
        calc_emissions_derv(parameters)
    elif type_run == "plots_analytic":
        plots_analytic(root = "2_sector_analytic",LOAD = 0, init_params = 2, scenario = 7)
        #plots_analytic(LOAD = 1, LOAD_filename = "results/2_sector_analytic_19_37_47__20_03_2024")
         
    else:
        raise ValueError("Wrong TYPE")

if __name__ == '__main__':
    type_run = "analytic"#"plots"#"plots_analytic"#"analytic"#,"calc_analytic"
    main(type_run)
