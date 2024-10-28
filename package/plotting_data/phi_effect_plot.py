
# imports
import matplotlib.pyplot as plt
from pyrsistent import v
from package.resources.utility import load_object, calc_bounds
from matplotlib.cm import  get_cmap

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
        ax.grid()
    fig.supxlabel(r"Social susceptibility, $\phi$", fontsize="12")
    axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.subplots_adjust(right=0.2)  # Adjust right margin to make space for legend

    fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize="10")

    #plt.tight_layout()
    

    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_xlog"
    fig.savefig(f + ".png", dpi=600, format="png", bbox_inches='tight')


def plot_var_emisisons_simple_xlog(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10, 4), sharey=True)  # Increased width

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
        #ax.set_yscale('log')
        #ax.set_xlabel( r"Social susceptability, $\phi$")
        ax.grid()
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Variance cumulative carbon emissions, Var(E)")
    fig.supxlabel(r"Social susceptibility, $\phi$", fontsize="12")
    #axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.subplots_adjust(right=0.2)  # Adjust right margin to make space for legend

    fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_var_xlog"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps") 


def plot_var_emisisons_simple_xlog_ylog(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10, 4), sharey=True)  # Increased width

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
        ax.set_yscale('log')
        #ax.set_xlabel(r"Social susceptability, $\phi$")
        ax.grid()
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Variance cumulative carbon emissions, Var(E)")
    fig.supxlabel(r"Social susceptibility, $\phi$", fontsize="12")
    #axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.subplots_adjust(right=0.2)  # Adjust right margin to make space for legend

    fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_var_xlog_ylog"
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
    #print("base_params", base_params)
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    property_varied = var_params["property_varied"]
    network_titles = ["Small-World", "Stochastic Block Model", "Scale-Free"]
    emissions_array = load_object(fileName + "/Data", "emissions_array")
    print(emissions_array.shape)
    scenarios_titles = [r"No carbon price, $\tau = 0$", r"Low carbon price, $\tau = 0.1$", r"High carbon price, $\tau = 1$"]
    scenarios_titles = [r"$\tau = 0$", r"$\tau = 0.1$", r"$\tau = 1$"]

    plot_emissions_simple_xlog(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    #plot_var_emisisons_simple_xlog_ylog(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    plot_var_emisisons_simple_xlog(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/phi_vary_17_08_30__24_10_2024"#phi_vary_16_44_12__22_08_2024",
    )
