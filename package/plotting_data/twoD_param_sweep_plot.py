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

def main(
    fileName = "results/splitting_eco_warriors_single_add_greens_17_44_05__01_02_2023",
    PLOT_TYPE = 2,
    dpi_save = 600,
    levels = 10,
    latex_bool = 0
) -> None:
        
    #variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    #results_emissions = load_object(fileName + "/Data", "results_emissions_stock")
    #key_param_array = load_object(fileName + "/Data","key_param_array")
    base_params = load_object(fileName + "/Data","base_params")
    print("base params",base_params)

    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    results_emissions = load_object(fileName + "/Data", "results_emissions_stock")
    #print(results_emissions.shape)
    #quit()
    #results_emissions
    #matrix_emissions = results_emissions.reshape((variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"]))

    matrix_emissions = np.mean(results_emissions, axis=2)
    #double_phase_diagram(fileName, matrix_emissions, r"Total normalised emissions $E/NM$", "emissions",variable_parameters_dict, get_cmap("Reds"),dpi_save, levels,latex_bool = latex_bool)  
    
    col_dict = variable_parameters_dict["col"]
    #print("col dict",col_dict)
    #col_dict["vals"] = col_dict["vals"][:-1]
    #print("col dict",col_dict)
    row_dict = variable_parameters_dict["row"]
    #print("row dict",row_dict)
    #quit()

    row_label = row_dict["title"]#r"Attitude Beta parameters, $(a,b)$"#r"Number of behaviours per agent, M"
    col_label = col_dict["title"]#r'Confirmation bias, $\theta$'
    y_label = "Emissions stock, $E/(NM)$"#col_dict["title"]#r"Identity variance, $\sigma^2$"
        
    #multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 1, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
    multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 0, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
    seed_reps_select = 4
    multi_line_matrix_plot_stacked(fileName,results_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 0, col_label, row_label, y_label,seed_reps_select)
    
    




    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/two_param_sweep_average_16_23_26__23_11_2023",
        PLOT_TYPE=2
    )