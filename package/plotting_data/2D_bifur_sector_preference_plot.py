"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import load_object

def plot_bifurcation_sectors(fileName, data_array,base_params,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=1, ncols=base_params["M"],sharey="row", constrained_layout = True, figsize= (10,6))

    #I need to sperate the data out so that its shape is: sector(M), one d thing im varying (Q?), the different people(N)[M,Q,N], currently its (Q,N,M)
    transposed_data = np.transpose(data_array, (2, 0, 1))

    
    for i, ax in enumerate(axes.flat):
        data_sector = transposed_data[i]
        for j in range(len(property_values_list)):
            x_vals = [property_values_list[j]]*(len(data_sector[j]))
            y_vals = data_sector[j]
            ax.plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)

        ax.set_ylabel("Final preference, $A_{\\tau,i,%s}$" % (str(i+1)))
        ax.set_xlabel(property_varied_title)
        ax.set_ylim(0,1)

    plotName = fileName + "/Prints"

    f = plotName + "/bifurcation_preferences_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_bifurcation_sectors_2d(fileName, data_array,base_params,property_values_list_1, property_varied_1, property_varied_title_1,property_values_list_2, property_varied_2, property_varied_title_2, dpi_save):

    fig, axes = plt.subplots(nrows=len(property_values_list_1), ncols=base_params["M"],sharey="row", constrained_layout = True, figsize= (10,6))

    #its currently [val_1,val_2,N,M] needs to go to [val_1, M,val_2,N]
    #(property_reps_1,property_reps_2, params["N"],params["M"])
    transposed_data = np.transpose(data_array, (0,3,1,2))

    for i, val_1 in enumerate(property_values_list_1):
        for j in range(base_params["M"]):
            data_sector = transposed_data[i][j]
            for k in range(len(property_values_list_2)):
                x_vals = [property_values_list_2[k]]*(len(data_sector[k]))
                y_vals = data_sector[k]
                axes[i][j].plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)

            #axes[i][j].set_ylabel("Final preference, $A_{\\tau,i,%s}$" % (str(j+1)))
            #axes[i][j].set_xlabel(property_varied_title_2)
            axes[i][j].set_ylim(0,1)

    low_carbon_substitutability_array = np.linspace(base_params["low_carbon_substitutability_lower"],base_params["low_carbon_substitutability_upper"], num=base_params["M"])
    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(low_carbon_substitutability_array[i],3))) for i in range(len(low_carbon_substitutability_array))]
    rows = ["%s=%s" % (property_varied_title_1,str(round(val,3))) for val in property_values_list_1]

    pad = 2 # in points

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='small', ha='right', va='center',rotation=90)
        
    fig.supxlabel("$%s$" % property_varied_title_2)
    fig.supylabel("Final preference, $A_{\\tau,i,m}$")

    plotName = fileName + "/Prints"

    f = plotName + "/2d_bifurcation_preferences_%s_%s" %(property_varied_1,property_varied_2)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_bifurcation_sectors_2d(fileName, data_array,base_params,property_values_list_1, property_varied_1, property_varied_title_1,property_values_list_2, property_varied_2, property_varied_title_2, dpi_save):

    if base_params["M"] == 1:
        fig, axes = plt.subplots(nrows=1, ncols=len(property_values_list_1),sharey=True, constrained_layout = True, figsize= (10,6))

        #its currently [val_1,val_2,N,M] needs to go to [val_1, M,val_2,N]
        #(property_reps_1,property_reps_2, params["N"],params["M"])
        transposed_data = np.transpose(data_array, (0,3,1,2))

        #print( transposed_data.shape)

        for i, val_1 in enumerate(property_values_list_1):
            axes[i].grid()
            data_sector = transposed_data[i][0]
            for k in range(len(property_values_list_2)):
                x_vals = [property_values_list_2[k]]*(len(data_sector[k]))
                y_vals = data_sector[k]
                axes[i].plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)
            axes[i].set_ylim(0,1)

        cols = ["%s=%s" % (property_varied_title_1,str(round(val,3))) for val in property_values_list_1]

        pad = 2 # in points

        for ax, col in zip(axes, cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='small', ha='center', va='baseline')
            
        fig.supxlabel("$%s$" % property_varied_title_2)
        fig.supylabel("Final preference, $A_{t_{max},i,1}$")
    else:
        fig, axes = plt.subplots(nrows=len(property_values_list_1), ncols=base_params["M"],sharey="row", constrained_layout = True, figsize= (10,6))

        #its currently [val_1,val_2,N,M] needs to go to [val_1, M,val_2,N]
        #(property_reps_1,property_reps_2, params["N"],params["M"])
        transposed_data = np.transpose(data_array, (0,3,1,2))

        for i, val_1 in enumerate(property_values_list_1):
            for j in range(base_params["M"]):
                data_sector = transposed_data[i][j]
                for k in range(len(property_values_list_2)):
                    x_vals = [property_values_list_2[k]]*(len(data_sector[k]))
                    y_vals = data_sector[k]
                    axes[i][j].plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)

                #axes[i][j].set_ylabel("Final preference, $A_{\\tau,i,%s}$" % (str(j+1)))
                #axes[i][j].set_xlabel(property_varied_title_2)
                axes[i][j].set_ylim(0,1)

        low_carbon_substitutability_array = np.linspace(base_params["low_carbon_substitutability_lower"],base_params["low_carbon_substitutability_upper"], num=base_params["M"])
        cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(low_carbon_substitutability_array[i],3))) for i in range(len(low_carbon_substitutability_array))]
        rows = ["%s=%s" % (property_varied_title_1,str(round(val,3))) for val in property_values_list_1]

        pad = 2 # in points

        for ax, col in zip(axes[0], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='small', ha='center', va='baseline')

        for ax, row in zip(axes[:,0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='small', ha='right', va='center',rotation=90)
            
        fig.supxlabel("$%s$" % property_varied_title_2)
        fig.supylabel("Final preference, $A_{\\tau,i,m}$")

    plotName = fileName + "/Prints"

    f = plotName + "/2d_bifurcation_preferences_%s_%s" %(property_varied_1,property_varied_2)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_bifurcation_sectors_2d_n_color(fileName, data_array,base_params,property_values_list_1, property_varied_1, property_varied_title_1,property_values_list_2, property_varied_2, property_varied_title_2, dpi_save):

    if base_params["M"] == 1:
        fig, axes = plt.subplots(nrows=1, ncols=len(property_values_list_1),sharey=True, constrained_layout = True, figsize= (10,6))

        #its currently [val_1,val_2,N,M] needs to go to [val_1, M, val_2,N]
        #(property_reps_1,property_reps_2, params["N"],params["M"])
        #transposed_data = np.transpose(data_array, (0,3,1,2))

        #its currently [val_1,val_2,N,M] needs to go to [val_1, M, N, val_2], for the lines being the same colour
        transposed_data = np.transpose(data_array, (0,3,2,1))

        #print( transposed_data.shape)

        for i, val_1 in enumerate(property_values_list_1):#loop different plots
            axes[i].grid()
            data_sector = transposed_data[i][0]#theres only 1 behaviour
            for k in range(base_params["N"]):#loop different
                x_vals = property_values_list_2
                y_vals = data_sector[k]
                axes[i].plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)
            axes[i].set_ylim(0,1)

        cols = ["$\sigma_1$=%s" % (str(round(val,3))) for val in property_values_list_1]

        pad = 2 # in points

        for ax, col in zip(axes, cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='small', ha='center', va='baseline')
            
        fig.supxlabel("$%s$" % property_varied_title_2)
        fig.supylabel("Final preference, $A_{t_{max},i,1}$")
    else:
        fig, axes = plt.subplots(nrows=len(property_values_list_1), ncols=base_params["M"],sharey="row", constrained_layout = True, figsize= (10,6))

        #its currently [val_1,val_2,N,M] needs to go to [val_1, M,val_2,N]
        #(property_reps_1,property_reps_2, params["N"],params["M"])
        #transposed_data = np.transpose(data_array, (0,3,1,2))

        #its currently [val_1,val_2,N,M] needs to go to [val_1, M, N, val_2], for the lines being the same colour
        transposed_data = np.transpose(data_array, (0,3,2,1))

        for i, val_1 in enumerate(property_values_list_1):
            for j in range(base_params["M"]):
                data_sector = transposed_data[i][j]
                for k in range(base_params["N"]):#loop different
                    x_vals = property_values_list_2
                    y_vals = data_sector[k]
                    axes[i][j].plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)

                #axes[i][j].set_ylabel("Final preference, $A_{\\tau,i,%s}$" % (str(j+1)))
                #axes[i][j].set_xlabel(property_varied_title_2)
                axes[i][j].set_ylim(0,1)

        low_carbon_substitutability_array = np.linspace(base_params["low_carbon_substitutability_lower"],base_params["low_carbon_substitutability_upper"], num=base_params["M"])
        cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(low_carbon_substitutability_array[i],3))) for i in range(len(low_carbon_substitutability_array))]
        rows = ["%s=%s" % (property_varied_title_1,str(round(val,3))) for val in property_values_list_1]

        pad = 2 # in points

        for ax, col in zip(axes[0], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='small', ha='center', va='baseline')

        for ax, row in zip(axes[:,0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='small', ha='right', va='center',rotation=90)
            
        fig.supxlabel("$%s$" % property_varied_title_2)
        fig.supylabel("Final preference, $A_{\\tau,i,m}$")

    plotName = fileName + "/Prints"

    f = plotName + "/n_color_2d_bifurcation_preferences_%s_%s" %(property_varied_1,property_varied_2)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_bifurcation_sectors_2d_alt(fileName, data_array,base_params,property_values_list_1, property_varied_1, property_varied_title_1,property_values_list_2, property_varied_2, property_varied_title_2,plot_var,plot_var_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(property_values_list_1), ncols=len(property_values_list_2),sharey="row", constrained_layout = True, figsize= (10,6))

    #its currently [val_1,val_2,N,M] needs to go to [val_1,val_2,M,N]
    #(property_reps_1,property_reps_2, params["N"],params["M"])
    transposed_data = np.transpose(data_array, (0,1,3,2))
    low_carbon_substitutability_array = np.linspace(base_params["low_carbon_substitutability_lower"],base_params["low_carbon_substitutability_upper"], num=base_params["M"])
    #low_carbon_substitutability_array = np.linspace(base_params["low_carbon_substitutability_lower"],base_params["low_carbon_substitutability_upper"], num=base_params["M"])
    #print("low_carbon_substitutability_array",low_carbon_substitutability_array)

    for i, val_1 in enumerate(property_values_list_1):
        for j in range(len(property_values_list_2)):
            data_sector = transposed_data[i][j]
            for k in range(base_params["M"]):
                #print("data_sector",data_sector, data_sector.shape)
                
                x_vals = [low_carbon_substitutability_array[k]]*(len(data_sector[k]))
                y_vals = data_sector[k]
                #print(len(x_vals ),len(y_vals))
                #quit()
                axes[i][j].plot(x_vals,y_vals, ls="", marker=".", linewidth = 0.5)

            #axes[i][j].set_ylabel("Final preference, $A_{\\tau,i,%s}$" % (str(j+1)))
            #axes[i][j].set_xlabel(property_varied_title_2)
            if plot_var == "preference":
                axes[i][j].set_ylim(0,1)

    
    #cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(low_carbon_substitutability_array[i],3))) for i in range(len(low_carbon_substitutability_array))]
    cols = ["%s=%s" % ("$%s$" % (property_varied_title_2),str(round(val,4))) for val in property_values_list_2]
    rows = ["%s=%s" % (property_varied_title_1,str(round(val,4))) for val in property_values_list_1]

    pad = 2 # in points

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='small', ha='right', va='center',rotation=90)
        
    #fig.supxlabel("$%s$" % property_varied_title_2)
    fig.supxlabel("Sector substitutability, $\sigma_m$")
    fig.supylabel(plot_var_title)

    plotName = fileName + "/Prints"

    f = plotName + "/2d_bifurcation_preferences_alt_%s_%s_%s" %(property_varied_1,property_varied_2, plot_var)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")



def main(
    fileName = "results/2D_bifur_one_param_sweep_20_31_14__02_11_2023",
    dpi_save = 600,
    latex_bool = 0,
    PLOT_TYPE = 1
    ) -> None: 

    ############################
    
    base_params = load_object(fileName + "/Data", "base_params")
    var_params_1  = load_object(fileName + "/Data" , "var_params_1")
    property_values_list_1 = load_object(fileName + "/Data", "property_values_list_1")
    var_params_2  = load_object(fileName + "/Data" , "var_params_2")
    property_values_list_2 = load_object(fileName + "/Data", "property_values_list_2")

    #print("property_values_list_1", property_values_list_1)
    #print("property_values_list_2", property_values_list_2)

    property_varied_1 = var_params_1["property_varied"]#"ratio_preference_or_consumption_state",
    property_min_1 = var_params_1["property_min"]#0,
    property_max_1 = var_params_1["property_max"]#1,
    property_reps_1 = var_params_1["property_reps"]#10,
    property_varied_title_1 = var_params_1["property_varied_title"]

    property_varied_2 = var_params_2["property_varied"]#"ratio_preference_or_consumption_state",
    property_min_2 = var_params_2["property_min"]#0,
    property_max_2 = var_params_2["property_max"]#1,
    property_reps_2 = var_params_2["property_reps"]#10,
    property_varied_title_2 = var_params_2["property_varied_title"]

    data_array = load_object(fileName + "/Data", "data_array")

    if PLOT_TYPE == 1:
        # look at splitting of the last behaviour with preference dissonance at final time step
        
        #plot_bifurcation_sectors_2d(fileName,data_array,base_params,property_values_list_1, property_varied_1, property_varied_title_1,property_values_list_2, property_varied_2, property_varied_title_2, dpi_save)
        plot_bifurcation_sectors_2d_n_color(fileName,data_array,base_params,property_values_list_1, property_varied_1, property_varied_title_1,property_values_list_2, property_varied_2, property_varied_title_2, dpi_save)
        #plot_var = "preference"
        #plot_var_title = "Final preference, $A_{t_{max},i,1}$"
        #plot_bifurcation_sectors_2d_alt(fileName,data_array,base_params,property_values_list_1, property_varied_1, property_varied_title_1,property_values_list_2, property_varied_2, property_varied_title_2,plot_var,plot_var_title, dpi_save)
    if PLOT_TYPE == 3:
        #print("property_varied_1, property_varied_title_1, property_varied_2, property_varied_title_2")
        #print(property_varied_1, property_varied_title_1, property_varied_2, property_varied_title_2)
        #quit()

        property_varied_title_1 = "sigma"
        property_varied_title_2 = "confirmation bias"

        property_varied_1 = "sigma"
        property_varied_2 = "confirmation_bias"


        data_array_H = load_object(fileName + "/Data", "data_array_H")
        data_array_L = load_object(fileName + "/Data", "data_array_L")
        
        plot_var = "preference"
        plot_var_title = "Final preference, $A_{t_{max},i,m}$"
        plot_bifurcation_sectors_2d_alt(fileName,data_array,base_params,property_values_list_1, property_varied_1, property_varied_title_1,property_values_list_2, property_varied_2, property_varied_title_2,plot_var,plot_var_title, dpi_save)
        
        plot_var = "H"
        plot_var_title = "Final high carbon quantity, $H_{t_{max},i,m}$"
        plot_bifurcation_sectors_2d_alt(fileName,data_array_H,base_params,property_values_list_1, property_varied_1, property_varied_title_1,property_values_list_2, property_varied_2, property_varied_title_2,plot_var,plot_var_title, dpi_save)
        plot_var = "L"
        plot_var_title = "Final low carbon quantity, $L_{t_{max},i,m}$"
        plot_bifurcation_sectors_2d_alt(fileName,data_array_L,base_params,property_values_list_1, property_varied_1, property_varied_title_1,property_values_list_2, property_varied_2, property_varied_title_2,plot_var,plot_var_title, dpi_save)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/2D_bifur_param_sweep_14_52_59__04_12_2023",
        PLOT_TYPE = 1
    )

