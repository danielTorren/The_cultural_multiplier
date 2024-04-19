import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from package.resources.utility import produce_name_datetime, save_object, createFolder, load_object
from sympy import symbols, diff, simplify, lambdify, print_latex, And, solve
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

def gen_1d_data(variable_parameters,parameters):
    variable_parameters["values"] = np.linspace(variable_parameters["min"], variable_parameters["max"], variable_parameters["reps"])

    parameters[variable_parameters["property"]] = variable_parameters["values"]

    return variable_parameters, parameters

def plot_contours(fileName,scen, B_values, tau_values, E_F_values, derivative_values, axis_val):
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

def plot_contours_2_scen(fileName,B_values, tau_values, E_F_values1, derivative_values1, E_F_values2, derivative_values2, axis_val):
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

def plot_sector_emissions(fileName,variable_parameters, data_E_F, data_E_F_1, data_E_F_2, scenario):
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(7, 7),constrained_layout = True)

    var1_values = variable_parameters["values"]
    var1_title = variable_parameters["title"]

    # Plot E_F
    ax.plot(var1_values, data_E_F, label = r'Emissions both sectors, $E_F$')
    ax.plot(var1_values, data_E_F_1, label = r'Emissions sector 1, $E_{F,1}$')
    ax.plot(var1_values, data_E_F_2, label = r'Emissions sector 2, $E_{F,2}$')
    ax.set_xlabel(var1_title)
    ax.set_ylabel(r'Emissions $E$')
    ax.legend()

    #plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/sector_emisisons" + variable_parameters["property"] 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

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

def run_plots(root,LOAD, init_params, scenario, LOAD_filename = "filename"):
    
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
        fileName, variable_parameters, parameters_run, scenario, init_params = set_up_data(root, init_params, scenario)
        
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
    if scenario not in (2,3):
        plot_contours_gen(fileName,variable_parameters, data_E_F, data_derv_E_F,levels, 1, scenario)
    plot_line_with_colorbar(fileName,variable_parameters, data_E_F, data_derv_E_F, 0, scenario)
    plot_line_with_colorbar(fileName, variable_parameters, data_E_F, data_derv_E_F, 1, scenario)

    #plot_contours(fileName, "Scenario 1",B_values, tau_values, data_E_F_1, data_derv_E_F_1, axis_val)
    #plot_contours(fileName, "Scenario 2",B_values, tau_values, data_E_F_2, data_derv_E_F_2, axis_val)
    #plot_contours_2_scen(fileName, B_values, tau_values, data_E_F_1, data_derv_E_F_1, data_E_F_2, data_derv_E_F_2, axis_val)

    plt.show()

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
    H_1 = (BD_sym*chi1_sym**nu_sym/Z_sym) + h1_sym
    partial_H1_tau1_sympy = diff(H_1, tau1_sym)

#########################################################################################################
    #H2
    #DERIVE IT SUING SYMPY
    H_2 = (BD_sym*chi2_sym**nu_sym/Z_sym) + h2_sym
    partial_H2_tau1_sympy = diff(H_2, tau1_sym)

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
                #"tau1": 1.2,#1,
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
                "gamma1": 1,#0.8,
                "gamma2": 1,#1.1
    }

        
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
    #SECTOR EMISSIONS
    EF1 = (BD_sym * (gamma1_sym*chi1_sym**nu_sym) / Z_sym) + gamma1_sym*h1_sym
    EF2 = (BD_sym * (gamma2_sym*chi2_sym**nu_sym) / Z_sym) + gamma2_sym*h2_sym

    derv_EF1 = diff(EF1, tau1_sym)
    derv_EF1_sub = derv_EF1.subs(subs_dict)

    derv_EF2 = diff(EF2, tau1_sym)
    

    ##########################################################################################################

    # Define EF equation with BD substituted
    EF = (BD_sym * (gamma1_sym*chi1_sym**nu_sym + gamma2_sym*chi2_sym**nu_sym) / Z_sym) + gamma1_sym*h1_sym + gamma2_sym*h2_sym
    derv_EF = diff(EF, tau1_sym)

    #manual_derv_EF = 
    #manual_derv_87  = gamma1_sym * BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * diff(chi1_sym, tau1_sym) - Z_sym**(-1) * gamma1_sym * h1_sym * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) * diff(Z_sym, tau1_sym)
    #manual_derv_87_subbed  = gamma1_sym * BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * manual_derv_chi1_sym - Z_sym**(-1) * gamma1_sym * h1_sym * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) * manual_derv_Z_sym
    
    manual_derv_EF_tau1 = gamma1_sym * BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * ((gamma1_sym*chi1_sym / (PBH1_sym + gamma1_sym*tau1_sym)) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) / (nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1)) - Z_sym**(-1) * gamma1_sym * h1_sym * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) * (((Omega1_sym * PL1_sym + PBH1_sym + gamma1_sym*tau1_sym) * gamma1_sym * nu_sym * chi1_sym**nu_sym) / (PBH1_sym + gamma1_sym*tau1_sym) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym))/(nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1) + gamma1_sym*chi1_sym**nu_sym * (PL1_sym * (sigma1_sym * Omega1_sym) / (PBH1_sym + gamma1_sym*tau1_sym) + 1))

    ##########################################################################################################

    print_latex(derv_EF1_sub)

    quit()
    return EF, manual_derv_EF_tau1

def inequality_solutions():

    # Define symbols
    tau1_sym, tau2_sym, B_sym, nu_sym, h1_sym, h2_sym, PBH1_sym, PBH2_sym, A1_sym, A2_sym, PL1_sym, PL2_sym, sigma1_sym, sigma2_sym, a1_sym, a2_sym, gamma1_sym, gamma2_sym = symbols('tau1 tau2 B nu h1 h2 PBH1 PBH2 A1 A2 PL1 PL2 sigma1 sigma2 a1 a2 gamma1 gamma2')
    
    # Define BD equation
    BD_sym = B_sym - h1_sym * (PBH1_sym + gamma1_sym*tau1_sym) - h2_sym * (PBH2_sym + gamma2_sym*tau2_sym)

    # Define Omega1 equation
    Omega1_sym = ((PBH1_sym + gamma1_sym*tau1_sym) * A1_sym / (PL1_sym * (1 - A1_sym)))**sigma1_sym

    # Define Omega2 equation
    Omega2_sym = ((PBH2_sym + tau2_sym) * A2_sym / (PL2_sym * (1 - A2_sym)))**sigma2_sym

    # Define chi1 equation
    chi1_sym = (a1_sym / (PBH1_sym + gamma1_sym*tau1_sym)) * ((A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) + (1 - A1_sym))**(((nu_sym - 1) * sigma1_sym) / (nu_sym * (sigma1_sym - 1)))

    # Define chi2 equation
    chi2_sym = (a2_sym / (PBH2_sym + gamma2_sym*tau2_sym)) * ((A2_sym * Omega2_sym**((sigma2_sym - 1) / sigma2_sym)) + (1 - A2_sym))**(((nu_sym - 1) * sigma2_sym) / (nu_sym * (sigma2_sym - 1)))

    # Define Z equation
    Z_sym = (chi1_sym**nu_sym * (Omega1_sym * PL1_sym + PBH1_sym + gamma1_sym*tau1_sym)) + (chi2_sym**nu_sym * (Omega2_sym * PL2_sym + PBH2_sym + gamma2_sym*tau2_sym))

    # Define EF equation with BD substituted
    #EF = (BD_sym * (gamma1_sym*chi1_sym**nu_sym + gamma2_sym*chi2_sym**nu_sym) / Z_sym) + gamma1_sym*h1_sym + gamma2_sym*h2_sym
    manual_derv_EF_tau1 = gamma1_sym * BD_sym * Z_sym**(-1) * nu_sym * chi1_sym**(nu_sym - 1) * ((gamma1_sym*chi1_sym / (PBH1_sym + gamma1_sym*tau1_sym)) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym)) / (nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1)) - Z_sym**(-1) * gamma1_sym * h1_sym * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) - BD_sym * Z_sym**(-2) * (gamma1_sym * chi1_sym**nu_sym + gamma2_sym * chi2_sym**nu_sym) * (((Omega1_sym * PL1_sym + PBH1_sym + gamma1_sym*tau1_sym) * gamma1_sym * nu_sym * chi1_sym**nu_sym) / (PBH1_sym + gamma1_sym*tau1_sym) * ((sigma1_sym * (nu_sym - 1) * A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym))/(nu_sym * (A1_sym * Omega1_sym**((sigma1_sym - 1) / sigma1_sym) + (1 - A1_sym))) - 1) + gamma1_sym*chi1_sym**nu_sym * (PL1_sym * (sigma1_sym * Omega1_sym) / (PBH1_sym + gamma1_sym*tau1_sym) + 1))
    print("manual_derv_EF_tau1", manual_derv_EF_tau1)

    # Define constraints
    constraints = And(B_sym >= h1_sym * (PBH1_sym + gamma1_sym*tau1_sym) - h2_sym * (PBH2_sym + gamma2_sym*tau2_sym), B_sym > 0, a1_sym + a2_sym == 1, 0 < a1_sym, a1_sym < 1, 0 < a2_sym, a2_sym < 1, gamma1_sym > 0, gamma2_sym> 0, nu_sym > 1,  sigma1_sym > 1, sigma2_sym > 1, 0 < A1_sym, A1_sym < 1, 0 < A2_sym, A2_sym < 1, PL1_sym > 0, PL2_sym > 0, PBH1_sym > 0, PBH2_sym > 0, tau1_sym>=0, tau2_sym>=0)
    print("constraints", constraints)

    subs_dict = {#BASE PARAMS
        "A1": 0.5,
        #"A2": 0.5,
        "tau1": 0,#1,
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
        "gamma1": 1,#0.8,
        "gamma2": 1,#1.1
    }

    substituted_eq = manual_derv_EF_tau1.subs(subs_dict)

    # Define inequalities
    inequality = substituted_eq > 0
    print("inequality", inequality)

    #(B_sym, a1_sym, a2_sym, gamma1_sym, gamma2_sym, nu_sym, sigma1_sym, sigma2_sym, A1_sym, A2_sym, PL1_sym, PL2_sym, PBH1_sym,  PBH2_sym, tau1_sym, tau2_sym)

    varied_params = [str(sym) for sym in manual_derv_EF_tau1.free_symbols]
    derv_EF_tau1_func = lambdify(varied_params,substituted_eq, 'numpy')
    epsilon = 1e-5
    A2_values = np.linspace(0+epsilon,1-epsilon,1000)
    subs_dict["A2"] = A2_values 

    #Calc values
    data_derv_EF_tau1 = derv_EF_tau1_func(**subs_dict)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5),constrained_layout = True)
    ax.plot(A2_values, data_derv_EF_tau1)
    ax.set_xlabel("A2")
    ax.set_ylabel("derivative E")
    ax.grid()
    plt.show()
    # Solve inequalities
    #solution = solve(inequality, (A2_sym),  domain=constraints)
    
    #print("SOLUTION", solution)

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

def plots_analytic(root = "2_sector_analytic",LOAD = 0, init_params = 4, scenario = 1, LOAD_filename = "filename"):
    
    if LOAD:
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
        
        #print("init_params, scenario",init_params, scenario)

        fileName, variable_parameters, parameters_run, scenario, init_params = set_up_data(root = root, init_params = init_params, scenario = scenario)



        EF, partial_derivative_EF_tau1, partial_derivative_EF_tau2, EF_tau1_tau2 = analytic_derivatives()


        #print(" EF",  EF)
        #print("partial_derivative_EF_tau1", partial_derivative_EF_tau1)
        #print("partial_derivative_EF_tau2",partial_derivative_EF_tau2) 


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

def set_up_data_1D(root = "2_sector_model", init_params = 4, scenario = 1):
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
                "B": 5,
                "gamma1":1,
                "gamma2":2
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
                "B": 5,
                "gamma1":1,
                "gamma2":2
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
                "B": 1,
                "gamma1":1,
                "gamma2":1
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
        case 6:
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
        case 7:
            parameters_1 = {
                "A1": 0.5,
                "A2": 0.1,
                "tau1": 0,
                "tau2": 0,
                "a1": 0.5,
                "a2": 0.5,
                "sigma1": 2,
                "sigma2": 2,
                "nu": 2,
                "PL1": 2,
                "PL2": 2,
                "PBH1": 2,
                "PBH2": 1,
                "h1": 0,
                "h2": 0,
                "B": 1,
                "gamma1":1,
                "gamma2":2
            }
        case 8:
            parameters_1 = {
                "A1": 0.5,
                "A2": 0.5,
                #"tau1": 0.1,
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
        case 2:
            variable_parameters = {
                "property": "A1",
                "min":0+1e-5,
                "max": 1-1e-5,
                "reps": 1000,
                "title": r'Sector 1 low carbon preference, $A_1$'
            }
        case 3:
            variable_parameters = {
                "property": "gamma1",
                "min":0+1e-5,
                "max": 1-1e-5,
                "reps": 1000,
                "title": r'Sector 1 emissions intensity, $\gamma_1$'
            }
        case 4:
            variable_parameters = {
                "property": "gamma2",
                "min":0+1e-5,
                "max": 1-1e-5,
                "reps": 1000,
                "title": r'Sector 2 emissions intensity, $\gamma_2$'
            }

    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    variable_parameters, parameters_run = gen_1d_data(variable_parameters, parameters_1)
    return fileName, variable_parameters, parameters_run, scenario, init_params

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

def calc_emissions_tax_rebound():

    #a = load_object("results/nu_tau_ratio_20_34_10__25_03_2024" + "/Data", "subs_dict")
    #print("a", a)

    #quit()

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

    #MESS AROUND WITH THE CASE OF FOOD AND TRAVEL? 
    """
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
    """
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
    print("SOLUTION FOUND")
    
    
    #######################################################################################################################################################
    root = "nu_tau_ratio"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    createFolder(fileName)


    #save_object(tau_C_func_1, fileName + "/Data", "tau_C_func_1")
    save_object(subs_dict, fileName + "/Data", "subs_dict")

    nu_plot = 0
    tau_plot = 0
    nu_tau_plot = 0
    together_plot = 1
    #######################################################################################################################################################
    # RUN THE Nu vs rebound plot
    if nu_plot:
        tau_C_func_1 = lambdify(nu_sym, tau_C_solution_no_nu_sub[1], 'numpy')#FOR SOME REASON YOU HAVE TO GET THE SECOND ONE???
        
        data_tau1_nu = []

        nu_values_lin = np.linspace(1.01, 10, 1000)
        #tau1_I_values_lin = np.linspace(0.1, 1.5, 5)
        for nu in nu_values_lin:
            tau_C_values_1 = tau_C_func_1(nu)
            data_tau1_nu.append(tau_C_values_1)

        save_object(nu_values_lin, fileName + "/Data", "nu_values_lin")
        save_object(data_tau1_nu, fileName + "/Data", "data_tau1_nu")
        
        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5),constrained_layout = True)
        ax1.plot(nu_values_lin, data_tau1_nu)
        ax1.set_xlabel(r"Inter-sector substitutability, $\nu$")
        ax1.set_ylabel(r"Rebound tax multiplier, $M_R = \frac{\tau_C}{\tau_I}$")
        ax1.grid()

        plt.tight_layout()

        plotName = fileName + "/Plots"
        f = plotName + "/incomplete"
        fig1.savefig(f + ".eps", dpi=600, format="eps")
        fig1.savefig(f + ".png", dpi=600, format="png")

    #######################################################################################################################################################
    # RUN THE TAU vs Rebound for differnet nu
    #print("tau_C_solution_no_nu_sub",tau_C_solution_no_nu_sub)
    if tau_plot:
        tau_C_func_1 = lambdify((nu_sym,tau1_I_sym), tau_C_solution_no_nu_sub[0], 'numpy')#FOR SOME REASON YOU HAVE TO GET THE SECOND ONE???
        data_tau1_tau = []

        nu_values_lin_tau = np.logspace(np.log10(1.01), 1, 5)
        tau1_I_values_lin = np.linspace(0.1, 1, 1000)
        for nu in nu_values_lin_tau:
            tau_C_values_1 = tau_C_func_1(nu, tau1_I_values_lin)
            data_tau1_tau.append(tau_C_values_1/tau1_I_values_lin)

        save_object(nu_values_lin_tau , fileName + "/Data", "nu_values_lin_tau")
        save_object(tau1_I_values_lin , fileName + "/Data", "tau1_I_values_lin ")
        save_object(data_tau1_tau, fileName + "/Data", "data_tau1_tau")

        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5),constrained_layout = True)
        
        for i,nu in enumerate(nu_values_lin_tau):
            ax2.plot(tau1_I_values_lin, data_tau1_tau[i], label= "$\\nu = %s$" % (round(nu,3)))
        #ax2.plot(nu_values_lin, tau_C_values_1, label= "Positive solutions")
        #ax.plot(nu_values, tau_C_values_0, label= "Negative solutions")
        ax2.set_xlabel(r"Sector 1 incomplete carbon tax,$\tau_I$ ")
        ax2.set_ylabel(r"Rebound tax multiplier, $M_R = \frac{\tau_C}{\tau_I}$")
        #ax2.set_title('linear')
        ax2.legend()
        ax2.grid()

        plt.tight_layout()

        plotName = fileName + "/Plots"
        f = plotName + "/tau_plot_incomplete"
        fig1.savefig(f + ".eps", dpi=600, format="eps")
        fig1.savefig(f + ".png", dpi=600, format="png")
    #######################################################################################################################################################
    # RUN THE TAU vs Rebound for differnet nu
    #print("tau_C_solution_no_nu_sub",tau_C_solution_no_nu_sub)
    if nu_tau_plot:
        tau_C_func_1 = lambdify((nu_sym,tau1_I_sym), tau_C_solution_no_nu_sub[0], 'numpy')#FOR SOME REASON YOU HAVE TO GET THE SECOND ONE???
        data_tau1_nu_tau = []

        nu_values_lin_nu_tau = np.linspace(1.01, 10, 1000)
        tau1_I_values_lin_nu_tau = np.linspace(0.01, 1, 5)
        for tau in tau1_I_values_lin_nu_tau:
            tau_C_values_1 = tau_C_func_1(nu_values_lin_nu_tau, tau)
            data_tau1_nu_tau.append(tau_C_values_1/tau)

        save_object(nu_values_lin_nu_tau , fileName + "/Data", "nu_values_lin_nu_tau")
        save_object(tau1_I_values_lin_nu_tau , fileName + "/Data", "tau1_I_values_lin_nu_tau")
        save_object(data_tau1_nu_tau, fileName + "/Data", "data_tau1_nu_tau")

        fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5),constrained_layout = True)
        
        for i,tau in enumerate(tau1_I_values_lin_nu_tau):
            ax3.plot(nu_values_lin_nu_tau , data_tau1_nu_tau[i], label= "$\\tau_I = %s$" % (round(tau,3)))
        ax3.set_xlabel(r"Inter-sector substitutability, $\nu$")
        ax3.set_ylabel(r"Rebound tax multiplier, $M_R = \frac{\tau_C}{\tau_I}$")
        ax3.legend()
        ax3.grid()

        plt.tight_layout()

        plotName = fileName + "/Plots"
        f = plotName + "/nu_plot_incomplete"
        fig1.savefig(f + ".eps", dpi=600, format="eps")
        fig1.savefig(f + ".png", dpi=600, format="png")
    
    #######################################################################################################################################################
    
    if together_plot:

        fig_together, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), constrained_layout=True)

        ax2 = axes[0]
        ax3 = axes[1]
        #I CANT BE BOTHERED TO RE-WRITE THIS ATM

        tau_C_func_1 = lambdify((nu_sym,tau1_I_sym), tau_C_solution_no_nu_sub[0], 'numpy')#FOR SOME REASON YOU HAVE TO GET THE SECOND ONE???

        #NU VARS

        data_tau1_tau = []

        nu_values_lin_tau = np.logspace(np.log10(1.01), 2, 5)
        tau1_I_values_lin = np.linspace(0.1, 1, 1000)
        for nu in nu_values_lin_tau:
            tau_C_values_1 = tau_C_func_1(nu, tau1_I_values_lin)
            data_tau1_tau.append( tau1_I_values_lin/tau_C_values_1)#1 -

        save_object(nu_values_lin_tau , fileName + "/Data", "nu_values_lin_tau")
        save_object(tau1_I_values_lin , fileName + "/Data", "tau1_I_values_lin ")
        save_object(data_tau1_tau, fileName + "/Data", "data_tau1_tau")

        #fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5),constrained_layout = True)
        
        for i,nu in enumerate(nu_values_lin_tau):
            ax2.plot(tau1_I_values_lin, data_tau1_tau[i], label= "$\\nu = %s$" % (round(nu,3)))
        #ax2.plot(nu_values_lin, tau_C_values_1, label= "Positive solutions")
        #ax.plot(nu_values, tau_C_values_0, label= "Negative solutions")
        ax2.set_xlabel(r"Sector 1 incomplete carbon tax,$\tau_I$ ")
        ax2.set_ylabel(r"Rebound tax multiplier, $M_R = \frac{\tau_I}{\tau_C}$")#1 - 
        #ax2.set_title('linear')
        ax2.legend()
        ax2.grid()

        # TAU VARS

        data_tau1_nu_tau = []

        nu_values_lin_nu_tau = np.linspace(1.01, 100, 1000)
        tau1_I_values_lin_nu_tau = np.linspace(0.01, 1, 5)
        for tau in tau1_I_values_lin_nu_tau:
            tau_C_values_1 = tau_C_func_1(nu_values_lin_nu_tau, tau)
            data_tau1_nu_tau.append(tau/tau_C_values_1)#1 -

        save_object(nu_values_lin_nu_tau , fileName + "/Data", "nu_values_lin_nu_tau")
        save_object(tau1_I_values_lin_nu_tau , fileName + "/Data", "tau1_I_values_lin_nu_tau")
        save_object(data_tau1_nu_tau, fileName + "/Data", "data_tau1_nu_tau")

        #fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5),constrained_layout = True)
        
        for i,tau in enumerate(tau1_I_values_lin_nu_tau):
            ax3.plot(nu_values_lin_nu_tau , data_tau1_nu_tau[i], label= "$\\tau_I = %s$" % (round(tau,3)))
        ax3.set_xlabel(r"Inter-sector substitutability, $\nu$")
        ax3.set_ylabel(r"Rebound tax multiplier, $M_R = \frac{\tau_C}{\tau_I}$")#1 - 
        ax3.legend()
        ax3.grid()
        
        plt.tight_layout()

        plotName = fileName + "/Plots"
        f = plotName + "/together_incomplete"
        fig_together.savefig(f + ".eps", dpi=600, format="eps")
        fig_together.savefig(f + ".png", dpi=600, format="png")
    #######################################################################################################################################################
    
    plt.show()


def main( type_run):

    if type_run == "plots":
        #run_plots(root = "2_sector_model", LOAD = 0, init_params = 4, scenario = 1)
        run_plots(LOAD = 1, LOAD_filename = "results/2_sector_analytic_19_37_47__20_03_2024")
    elif type_run == "plots_1D":
        run_plots_1D(root = "2_sector_model", LOAD = 0, init_params = 4, scenario = 1)#init_params = 8, scenario = 4
    elif type_run == "ineq":
        print("INSIDE")
        inequality_solutions()
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
    elif type_run == "rebound_tax":
        calc_emissions_tax_rebound()
    else:
        raise ValueError("Wrong TYPE")

if __name__ == '__main__':
    type_run = "rebound_tax"#"rebound_tax"#"plots"#"ineq", "plots_analytic"#"analytic"#,"calc_analytic"
    main(type_run)
