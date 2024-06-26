"""Plot multiple simulations varying two parameters
Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from package.resources.plot import (
    double_phase_diagram,
    multi_line_matrix_plot,
    multi_line_matrix_plot_stoch,
    multi_line_matrix_plot_stoch_bands,
    multiline
)
from package.resources.utility import (
    load_object
)
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

def multi_line_matrix_plot_stacked(    
        fileName, Z_array, col_vals, row_vals,  Y_param, cmap, dpi_save, col_axis_x, col_label, row_label, y_label, seed_reps_select
    ):

    Z_array_T = np.transpose(Z_array,(2,0,1))#put seeds at front

    fig, axes = plt.subplots(ncols = seed_reps_select,sharey=True,constrained_layout=True,figsize=(10, 5))#

    #print( Z_array_T[0][0],Z_array_T[1][0])
    #quit()
    for i, ax in enumerate(axes.flat):
        Z = Z_array_T[i]
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
        ax.grid()
        lc = multiline(xs, ys, c, ax= ax, cmap=cmap, lw=2)#xs, ys, c, ax=None, **kwargs
        
        if col_axis_x:
            ax.set_xlabel(col_label)#(r'Confirmation bias, $\theta$')

        else:
            ax.set_xlabel(row_label)#(r"Number of behaviours per agent, M")

    axcb = fig.colorbar(lc)
    if col_axis_x:
        axcb.set_label(row_label)#(r"Number of behaviours per agent, M")
    else:
        axcb.set_label(col_label)#)(r'Confirmation bias, $\theta$')

    

    plotName = fileName + "/Plots"
    f = plotName + "/stacked_multi_line_matrix_plot_%s_%s" % (Y_param, col_axis_x)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")  

def multi_line_matrix_plot_stoch_bands(
    fileName, Z_array, col_vals, row_vals,  Y_param, cmap, dpi_save, col_axis_x, col_label, row_label, y_label
    ):
    
    fig, ax = plt.subplots( constrained_layout=True)#figsize=(14, 7)
    
    
    #cmap = plt.get_cmap("cividis")

    if col_axis_x:#= 1
        c = Normalize()(row_vals)
        for i in range(len(Z_array)):
            data = Z_array_T[i]#(sigma, seeds)
            ys_mean = data.mean(axis=1)
            ys_min = data.min(axis=1)
            ys_max= data.max(axis=1)

            ax.plot(col_vals, ys_mean, ls="-", linewidth = 0.5, color = cmap(c[i]))
            ax.fill_between(col_vals, ys_min, ys_max, facecolor=cmap(c[i]), alpha=0.5)
        cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( row_vals), vmax=max(row_vals))), ax=ax
    )
    else:
        Z_array_T = np.transpose(Z_array,(1,0,2))#put (sigma, tau,seeds) from (tau,sigma, seeds)
        c = Normalize()(col_vals)
        for i in range(len(Z_array_T)):#loop through sigma
            data = Z_array_T[i]
            ys_mean = data.mean(axis=1)
            ys_min = data.min(axis=1)
            ys_max= data.max(axis=1)

            ax.plot(row_vals, ys_mean, ls="-", linewidth = 0.5,color=cmap(c[i]))
            ax.fill_between(row_vals, ys_min, ys_max, facecolor=cmap(c[i]), alpha=0.5)

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( col_vals), vmax=max(col_vals))), ax=ax
        )   

    #quit()
    ax.set_ylabel(y_label)#(r"First behaviour attitude variance, $\sigma^2$")


    
    if col_axis_x:
        cbar.set_label(row_label)#(r"Number of behaviours per agent, M")
        ax.set_xlabel(col_label)#(r'Confirmation bias, $\theta$')
    else:
        cbar.set_label(col_label)#)(r'Confirmation bias, $\theta$')
        ax.set_xlabel(row_label)#(r"Number of behaviours per agent, M")

    plotName = fileName + "/Plots"
    f = plotName + "/multi_line_matrix_plot_stoch_fill_%s_%s" % (Y_param, col_axis_x)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")  

def multi_line_matrix_plot_stoch_bands_inset(
    fileName, Z_array, col_vals, row_vals,  Y_param, cmap, dpi_save, col_axis_x, col_label, row_label, y_label
    ):
    
    fig, ax = plt.subplots( constrained_layout=True)#figsize=(14, 7)
    inset_ax = ax.inset_axes([0.4, 0.4, 0.55, 0.55])
    
    #cmap = plt.get_cmap("cividis")

    if col_axis_x:#= 1
        c = Normalize()(row_vals)
        for i in range(len(Z_array)):
            data = Z_array_T[i]#(sigma, seeds)
            ys_mean = data.mean(axis=1)
            ys_min = data.min(axis=1)
            ys_max= data.max(axis=1)

            ax.plot(col_vals, ys_mean, ls="-", linewidth = 0.5, color = cmap(c[i]))
            ax.fill_between(col_vals, ys_min, ys_max, facecolor=cmap(c[i]), alpha=0.5)

            inset_ax.plot(col_vals[-10:-1], ys_mean[-10:-1])
            inset_ax.fill_between(col_vals[-10:-1], ys_min[-10:-1], ys_max[-10:-1], facecolor=cmap(c[i]), alpha=0.5)

        cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( row_vals), vmax=max(row_vals))), ax=ax
    )
    else:
        Z_array_T = np.transpose(Z_array,(1,0,2))#put (sigma, tau,seeds) from (tau,sigma, seeds)
        c = Normalize()(col_vals)
        for i in range(len(Z_array_T)):#loop through sigma
            data = Z_array_T[i]
            ys_mean = data.mean(axis=1)
            ys_min = data.min(axis=1)
            ys_max= data.max(axis=1)

            ax.plot(row_vals, ys_mean, ls="-", linewidth = 0.5,color=cmap(c[i]))
            ax.fill_between(row_vals, ys_min, ys_max, facecolor=cmap(c[i]), alpha=0.5)
            inset_ax.plot(row_vals[-10:-1], ys_mean[-10:-1])
            inset_ax.fill_between(row_vals[-10:-1], ys_min[-10:-1], ys_max[-10:-1], facecolor=cmap(c[i]), alpha=0.5)
            
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(vmin=min( col_vals), vmax=max(col_vals))), ax=ax
        )   

    #quit()
    ax.set_ylabel(y_label)#(r"First behaviour attitude variance, $\sigma^2$")


    
    if col_axis_x:
        cbar.set_label(row_label)#(r"Number of behaviours per agent, M")
        ax.set_xlabel(col_label)#(r'Confirmation bias, $\theta$')
    else:
        cbar.set_label(col_label)#)(r'Confirmation bias, $\theta$')
        ax.set_xlabel(row_label)#(r"Number of behaviours per agent, M")

    plotName = fileName + "/Plots"
    f = plotName + "/multi_line_matrix_plot_stoch_fill_%s_%s" % (Y_param, col_axis_x)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png") 

def extremes_plot(fileName, Z_array,col_vals, row_vals):
    
    fig, ax = plt.subplots(constrained_layout=True)#figsize=(14, 7)
    #print("Z_array.shape", Z_array.shape)
    #quit()
    Z_array_T = np.transpose(Z_array,(1,0,2))#put (sigma, tau,seeds) from (tau,sigma, seeds)
    print("Z_array_T",Z_array_T.shape)
    #quit()
    #min
    data = Z_array_T[0]
    ys_mean = data.mean(axis=1)
    ys_min = data.min(axis=1)
    ys_max= data.max(axis=1)
    ax.plot(row_vals, ys_mean, ls="-", linewidth = 0.5,label ="Block 2 $\\sigma$ = %s" % (col_vals[0]) )
    ax.fill_between(row_vals, ys_min, ys_max, alpha=0.5)

    #max
    data = Z_array_T[-1]
    ys_mean = data.mean(axis=1)
    ys_min = data.min(axis=1)
    ys_max= data.max(axis=1)
    ax.plot(row_vals, ys_mean, ls="-", linewidth = 0.5,label ="Block 2 $\\sigma$ = %s" % (col_vals[-1]))
    ax.fill_between(row_vals, ys_min, ys_max , alpha=0.5)

    ax.legend()
    ax.set_xlabel("Carbon price")
    ax.set_ylabel("Emissions")
    plotName = fileName + "/Plots"
    f = plotName + "/extremes"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")  


def main(
    fileName = "results/splitting_eco_warriors_single_add_greens_17_44_05__01_02_2023",
    PLOT_TYPE = 2,
    dpi_save = 600,
    levels = 30,
    latex_bool = 0
) -> None:
        
    #variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    #results_emissions = load_object(fileName + "/Data", "results_emissions_stock")
    #key_param_array = load_object(fileName + "/Data","key_param_array")
    base_params = load_object(fileName + "/Data","base_params")
    print("base params",base_params)

    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    #variable_parameters_dict["title"] = "Block 2 substitutability"
    results_emissions = load_object(fileName + "/Data", "results_emissions_stock")
    print("variable_parameters_dict",variable_parameters_dict)
    #quit()

    #print(results_emissions.shape)
    #quit()
    #results_emissions
    #matrix_emissions = results_emissions.reshape((variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"]))

    matrix_emissions = np.mean(results_emissions, axis=2)
    #double_phase_diagram(fileName, matrix_emissions, r"Cumulative emissions $E$", "emissions",variable_parameters_dict, get_cmap("Reds"),dpi_save, levels,latex_bool = latex_bool)  
    
    col_dict = variable_parameters_dict["col"]
    #print("col dict",col_dict)
    #col_dict["vals"] = col_dict["vals"][:-1]
    #print("col dict",col_dict)
    row_dict = variable_parameters_dict["row"]
    #print("row dict",row_dict)
    #quit()

    row_label = row_dict["title"]#r"Attitude Beta parameters, $(a,b)$"#r"Number of behaviours per agent, M"
    col_label = col_dict["title"]#r'Confirmation bias, $\theta$'
    y_label = "Emissions stock, $E$"#col_dict["title"]#r"Identity variance, $\sigma^2$"
        
    #multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 1, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
    #multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 0, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
    #multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 1, col_label, row_label, y_label)
    seed_reps_select = 3
    #multi_line_matrix_plot_stacked(fileName,results_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 0, col_label, row_label, y_label,seed_reps_select)
    

    #multi_line_matrix_plot_stoch_bands(fileName, results_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"), dpi_save, 0, col_label, row_label, y_label)
    multi_line_matrix_plot_stoch_bands_inset(fileName, results_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"), dpi_save, 0, col_label, row_label, y_label)
        
    #extremes_plot(fileName, results_emissions, col_dict["vals"], row_dict["vals"])




    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/two_param_sweep_21_40_32__30_01_2024",
    )