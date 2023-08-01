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
    multi_line_matrix_plot_stoch_bands
)
from package.resources.utility import (
    load_object
)
import numpy as np

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

    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    results_emissions = load_object(fileName + "/Data", "results_emissions_stock")

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
        
    multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 1, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label

    

    """
    print("key_param_array ",key_param_array)
    #print("results_emissions[0]",results_emissions[0])
    #print("results_emissions[:][0][:]",results_emissions[:][0][:])
    print("results_emissions",results_emissions)
    print("mean?", results_emissions.mean(axis = 2))
    print(np.mean([0.30240306, 0.09407981, 0.09057053]))
    #quit()
    """

    #ys_mean = results_emissions[i].mean(axis=1)
    """
    if PLOT_TYPE == 2:

        col_dict = variable_parameters_dict["col"]
        row_dict = variable_parameters_dict["row"]

        row_label = row_dict["title"]#r"Attitude Beta parameters, $(a,b)$"#r"Number of behaviours per agent, M"
        col_label = col_dict["title"]#r'Confirmation bias, $\theta$'
        y_label = "Emissions stock, $E/(NM)$"#col_dict["title"]#r"Identity variance, $\sigma^2$"
        
        #print("variable_parameters_dict",variable_parameters_dict)
        #print("results_emissions", results_emissions)
        #quit()
                            #fileName, Z, col_vals, row_vals,  Y_param, cmap, dpi_save, col_axis_x, col_label, row_label, y_label
        #multi_line_matrix_plot_stoch(fileName, results_emissions, col_dict["vals"], row_dict["vals"], "emissions", get_cmap("plasma"),dpi_save, 1, col_label, row_label, y_label)
        multi_line_matrix_plot_stoch_bands(fileName,results_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 1, col_label, row_label, y_label)
        multi_line_matrix_plot_stoch_scatter(fileName,results_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 1, col_label, row_label, y_label)
    else:
        matrix_emissions = results_emissions.reshape((variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"]))
        double_phase_diagram(fileName, matrix_emissions, r"Stock emissions $E$", "emissions",variable_parameters_dict, get_cmap("Reds"),dpi_save, levels,latex_bool = latex_bool)  

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
            
        multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save, 0, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
        #####################################################################
    """
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/two_param_sweep_average_21_43_39__25_07_2023",
        PLOT_TYPE=2
    )