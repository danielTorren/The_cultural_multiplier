
# imports
import matplotlib.pyplot as plt
from pyrsistent import v
from package.resources.utility import load_object, calc_bounds
from matplotlib.cm import  get_cmap

def plot_emisisons_simple(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8), sharey=True)#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions = Data.mean(axis=1)
            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i])
            data_trans = Data.T
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i], alpha = 0.1)


        #ax.legend()
        ax.set_xlabel(r"Social susceptability, $\phi$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")

def plot_emissions_simple_xlog(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10, 4), sharey=True)  # Increased width

    for k, ax in enumerate(axes.flat):
        emissions = emissions_networks[k]

        for i in range(len(emissions)):
            Data = emissions[i]
            mu_emissions = Data.mean(axis=1)
            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c=colors_scenarios[i])
            data_trans = Data.T
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color=colors_scenarios[i], alpha=0.1)

        ax.set_xscale('log')
        #ax.set_xlabel(r"Social susceptibility, $\phi$")
        ax.set_title(network_titles[k], fontsize="12")
    
    fig.supxlabel(r"Social susceptibility, $\phi$", fontsize="12")
    axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.subplots_adjust(right=0.2)  # Adjust right margin to make space for legend

    fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize="10")

    #plt.tight_layout()
    

    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_xlog"
    fig.savefig(f + ".png", dpi=600, format="png", bbox_inches='tight')

def plot_var_emisisons_simple(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            var_emissions = Data.var(axis=1)
            ax.plot(property_vals[1:], var_emissions[1:], label=scenarios_titles[i], c = colors_scenarios[i])

        #ax.legend()
        ax.set_xlabel(r"Social susceptability, $\phi$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Variance cumulative carbon emissions, Var(E)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_var"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps")  

def plot_var_emisisons_simple_log(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            var_emissions = Data.var(axis=1)
            ax.plot(property_vals, var_emissions, label=scenarios_titles[i], c = colors_scenarios[i])

        #ax.legend()
        ax.set_yscale('log')
        ax.set_xlabel(r"Social susceptability, $\phi$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Variance cumulative carbon emissions, Var(E)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_var_log"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_var_emisisons_simple_xlog(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#

    #colors = iter(rainbow(np.linspace(0, 1,len(emissions_networks[0]))))

    for k, ax in enumerate(axes.flat):
        emissions  = emissions_networks[k]

        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            var_emissions = Data.var(axis=1)
            ax.plot(property_vals, var_emissions, label=scenarios_titles[i], c = colors_scenarios[i])

        #ax.legend()
        ax.set_xscale('log')
        ax.set_xlabel(r"Social susceptability, $\phi$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Variance cumulative carbon emissions, Var(E)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_var_xlog"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_M(
    fileName, M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#


    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)

            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i+1])
            data_trans = Data.T
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i+1], alpha = 0.1)
        ax.set_xlabel(r"Social susceptability, $\phi$")
        ax.set_title (network_titles[k])
        ax.set_ylim(-1,2)
    axes[0].set_ylabel(r"Carbon tax reduction, M")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_m_phi"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps")    

def plot_M_xlog(
    fileName, M_networks, scenarios_titles, property_vals, network_titles, colors_scenarios
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols = 3, nrows = 1,figsize=(15,8))#


    for k, ax in enumerate(axes.flat):
        emissions  = M_networks[k]
        
        for i in range(len(emissions)):
            #color = next(colors)#set color for whole scenario?
            Data = emissions[i]
            #print("Data", Data.shape)
            mu_emissions, __, __ = calc_bounds(Data, 0.95)

            ax.plot(property_vals, mu_emissions, label=scenarios_titles[i], c = colors_scenarios[i+1])

            data_trans = Data.T
            for v in range(len(data_trans)):
                ax.plot(property_vals, data_trans[v], color = colors_scenarios[i+1], alpha = 0.1)
        ax.set_ylim(-1,2)
        ax.set_xscale('log')
        ax.set_xlabel(r"Social susceptability, $\phi$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Carbon tax reduction, M")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_plot_m_phi_xlog"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    fig.savefig(f+ ".eps", dpi=600, format="eps")    


def main(
    fileName
    ) -> None: 
    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list
    ############################
    #BA
    base_params = load_object(fileName + "/Data", "base_params")
    var_params = load_object(fileName + "/Data" , "var_params")
    #print("base_params", base_params)
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    property_varied = var_params["property_varied"]
    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    emissions_array = load_object(fileName + "/Data", "emissions_array")
    scenarios_titles = [r"No carbon price, $\tau = 0$", r"Low carbon price, $\tau = 0.1$", r"High carbon price, $\tau = 1$"]
    scenarios_titles = [r"$\tau = 0$", r"$\tau = 0.1$", r"$\tau = 1$"]

    #plot_emisisons_simple(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    plot_emissions_simple_xlog(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    #plot_var_emisisons_simple(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    #plot_var_emisisons_simple_log(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    #plot_var_emisisons_simple_xlog(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    
    """
    #######################################
    static_tau_matrix = load_object(fileName + "/Data" , "tau_matrix")
    static_emissions_matrix = load_object(fileName + "/Data" , "emissions_matrix")

    print(static_tau_matrix.shape, static_emissions_matrix.shape)
    print(static_tau_matrix[0])
    print(static_emissions_matrix[0])

    quit()

    matrix1 = static_tau_matrix
    matrix2 = static_emissions_matrix
    # Create a figure and axis for the plots
    fig, ax = plt.subplots()

    # Plot each row where matrix1 is x and matrix2 is y
    for i in range(matrix1.shape[0]):
        ax.plot(matrix1[i], matrix2[i], alpha=0.5)

    # Adding labels and title
    ax.set_xlabel('tau')
    ax.set_ylabel('E')
    plt.show()
    quit()

    M_networks = load_object(fileName + "/Data" , "M_vals_networks")
    plot_M(fileName, M_networks, scenarios_titles, property_values_list, network_titles, colors_scenarios)
    plot_M_xlog(fileName, M_networks, scenarios_titles, property_values_list, network_titles, colors_scenarios)
    """
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/phi_vary_09_14_29__17_05_2024",
    )
