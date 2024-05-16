
# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
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

def plot_emisisons_simple_xlog(
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

        ax.set_xscale('log')
        #ax.legend()
        ax.set_xlabel(r"Social susceptability, $\phi$")
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Cumulative carbon emissions, E")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios_titles), fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_xlog"
    fig.savefig(f+ ".png", dpi=600, format="png")   

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
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    property_varied = var_params["property_varied"]
    network_titles = ["Watt-Strogatz Small-World", "Stochastic Block Model", "Barabasi-Albert Scale-Free"]
    emissions_array = load_object(fileName + "/Data", "emissions_array")
    scenarios_titles = labels_SW = [r"No carbon price, $\tau = 0$", r"Low carbon price, $\tau = 0.1$", r"High carbon price, $\tau = 1$"]

    #plot_emisisons_simple(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    plot_emisisons_simple_xlog(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    #plot_var_emisisons_simple(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    #plot_var_emisisons_simple_log(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    plot_var_emisisons_simple_xlog(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/phi_vary_12_00_50__16_05_2024",
    )
