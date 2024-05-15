
# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
import numpy as np

def plot_end_points_emissions_multi_BA_SBM_SW(
    fileName: str, Data_arr_BA, Data_arr_SBM, Data_arr_SW, property_title, property_save, property_vals, labels_BA, labels_SBM, labels_SW,seed_reps
):
    #nrows=seed_reps
    fig, axes = plt.subplots(nrows=3, ncols=Data_arr_BA.shape[0], figsize=(10, 6), constrained_layout=True)
    
    for j, Data_list in enumerate(Data_arr_BA):
        data_BA = Data_arr_BA[j].T
        data_SBM = Data_arr_SBM[j].T
        data_SW = Data_arr_SW[j].T
        for i in range(seed_reps):#loop through seeds

            #for i, ax in enumerate(axes.flat):
            axes[0][j].plot(property_vals, data_BA[i])
            axes[1][j].plot(property_vals, data_SBM[i], linestyle="dashed")
            axes[2][j].plot(property_vals, data_SW[i], linestyle="dotted")

        axes[0][j].set_title(labels_BA[j])
        axes[1][j].set_title(labels_SBM[j])
        axes[2][j].set_title(labels_SW[j])

        axes[0][j].grid()
        axes[1][j].grid()
        axes[2][j].grid()

        #axes[0][j].legend()
        #axes[1][j].legend()
        #axes[2][j].legend()
    
    fig.supxlabel(property_title)
    fig.supylabel(r"Cumulative emissions, E")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_BA_SBM_SW_seeds_" + property_save
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_emisisons_simple(
    fileName, emissions_networks, scenarios_titles, property_vals, network_titles
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]

        mu_emissions = emissions.mean(axis=1)
        ax.plot(property_vals, mu_emissions)
        data_trans = mu_emissions.T
        for v in range(len(data_trans)):
            ax.plot(property_vals, data_trans[v], alpha = 0.1)
        #ax.legend()
        ax.set_xlabel(r"Social susceptibility, $\phi$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")  

def main(
    fileName
    ) -> None: 

    ############################
    #BA
    base_params_BA = load_object(fileName + "/Data", "base_params_BA")
    var_params = load_object(fileName + "/Data" , "var_params")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    property_varied = var_params["property_varied"]

    emissions_array_BA = load_object(fileName + "/Data", "emissions_array_BA")
    labels_BA = [r"BA, No carbon price, $\tau = 0$", r"BA, Low carbon price, $\tau = 0.1$", r"BA, High carbon price, $\tau = 1$"]
    
    #base_params_SBM = load_object(fileName + "/Data", "base_params_SBM")

    emissions_array_SBM = load_object(fileName + "/Data", "emissions_array_SBM")
    labels_SBM = [r"SBM, No carbon price, $\tau = 0$", r"SBM, Low carbon price, $\tau = 0.1$", r"SBM, High carbon price, $\tau = 1$"]

    #base_params_SW = load_object(fileName + "/Data", "base_params_SW")

    emissions_array_SW = load_object(fileName + "/Data", "emissions_array_SW")
    labels_SW = [r"SW, No carbon price, $\tau = 0$", r"SW, Low carbon price, $\tau = 0.1$", r"SW, High carbon price, $\tau = 1$"]

    seed_reps = base_params_BA["seed_reps"]

    plot_end_points_emissions_multi_BA_SBM_SW(fileName, emissions_array_BA, emissions_array_SBM, emissions_array_SW, r"Social susceptability, $\phi$", property_varied, property_values_list, labels_BA, labels_SBM, labels_SW,seed_reps)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/BA_SBM_SW_phi_vary_10_40_18__26_01_2024",
    )
