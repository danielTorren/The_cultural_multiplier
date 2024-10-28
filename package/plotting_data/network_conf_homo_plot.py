
# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object, calc_bounds
from matplotlib.cm import  get_cmap
import numpy as np

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
        #ax.set_xlabel(r"Confirmation bias, $\theta$")
        ax.set_title(network_titles[k], fontsize="12")
    
    fig.supxlabel(r"Confirmation bias, $\theta$", fontsize="12")
    axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.subplots_adjust(right=0.2)  # Adjust right margin to make space for legend

    fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize="10")

    #plt.tight_layout()
    

    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_xlog"
    fig.savefig(f + ".png", dpi=600, format="png", bbox_inches='tight')




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
        
        ax.set_title (network_titles[k])
    axes[0].set_ylabel(r"Variance cumulative carbon emissions, Var(E)")
    fig.supxlabel(r"Confirmation bias, $\theta$", fontsize="12")
    #axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.subplots_adjust(right=0.2)  # Adjust right margin to make space for legend

    fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize="10")

    # plt.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_var_xlog"
    fig.savefig(f+ ".png", dpi=600, format="png") 
    #fig.savefig(f+ ".eps", dpi=600, format="eps") 

def plot_emissions_simple_homo(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):
    fig, axes = plt.subplots(ncols=len(emissions_networks), nrows=len(emissions_networks[0]), figsize=(15, 8), sharex=True)  # Increased width

    for i, network_em in enumerate(emissions_networks):
        for j, homo_em in enumerate(network_em): 
            for k in range(len(homo_em)):
                Data = homo_em[k]
                mu_emissions = Data.mean(axis=1)
                axes[j][i].plot(property_vals, mu_emissions, label=scenarios_titles[k], c=colors_scenarios[k])
                data_trans = Data.T
                for v in range(len(data_trans)):
                    axes[j][i].plot(property_vals, data_trans[v], color=colors_scenarios[k], alpha=0.1)

        #axes[j][i].set_xscale('log')
        #axes[j][i].set_ylabel(r"Homophily, $\phi$")
        axes[0][i].set_title(network_titles[i], fontsize="12")
    
    #DEAL WITH y labels
    homo_title_list = [r"No homophily", r"Low homophily", r"High homophily"]
    for i, homo_title in enumerate(homo_title_list):
        axes[i][0].set_ylabel(homo_title)

    heg_list = [r"No homophily", r"Low-carbon hegemony", r"High-carbon hegemony"]
    for i, heg_title in enumerate(heg_list):
        axes[i][2].set_ylabel(heg_title)

    fig.supxlabel(r"Confirmation bias, $\theta$", fontsize="12")
    fig.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    #axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    handles, labels = axes[0][0].get_legend_handles_labels()
    
    #fig.subplots_adjust(right=0.2)  # Adjust right margin to make space for legend

    fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize="10")

    #plt.tight_layout()
    

    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_conf"
    fig.savefig(f + ".png", dpi=600, format="png", bbox_inches='tight')

def plot_emissions_simple_xlog_homo_confidence(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):
    fig, axes = plt.subplots(ncols=len(emissions_networks), nrows=len(emissions_networks[0]), figsize=(15, 8), sharex=True)  # Increased width

    for i, network_em in enumerate(emissions_networks):
        for j, homo_em in enumerate(network_em): 
            for k in range(len(homo_em)):
                Data = homo_em[k]
                mu_emissions = Data.mean(axis=1)
                axes[j][i].plot(property_vals, mu_emissions, label=scenarios_titles[k], c=colors_scenarios[k])
                mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
                # Plot the 95% confidence interval as a shaded area
                axes[j][i].fill_between(property_vals, lower_bound, upper_bound, color=colors_scenarios[k], alpha=0.3)

        axes[j][i].set_xscale('log')
        #axes[j][i].set_ylabel(r"Homophily, $\phi$")
        axes[0][i].set_title(network_titles[i], fontsize="12")
    
    #DEAL WITH y labels
    homo_title_list = [r"No homophily", r"Low homophily", r"High homophily"]
    for i, homo_title in enumerate(homo_title_list):
        axes[i][0].set_ylabel(homo_title)

    heg_list = [r"No homophily", r"Low-carbon hegemony", r"High-carbon hegemony"]
    for i, heg_title in enumerate(heg_list):
        axes[i][2].set_ylabel(heg_title)

    fig.supxlabel(r"Confirmation bias, $\theta$", fontsize="12")
    fig.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    #axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    handles, labels = axes[0][0].get_legend_handles_labels()
    
    #fig.subplots_adjust(right=0.2)  # Adjust right margin to make space for legend

    fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize="10")

    #plt.tight_layout()
    

    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_xlog_confidence"
    fig.savefig(f + ".png", dpi=600, format="png", bbox_inches='tight')


def plot_emissions_simple_xlog_homo_confidence_invert(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):
    
    fig, axes = plt.subplots(ncols=len(emissions_networks), nrows=len(emissions_networks[0]), figsize=(15, 8), sharex=True)  # Increased width
    
    homo_title_list = [r"No homophily", r"Low homophily", r"High homophily"]
    heg_list = [r"No homophily", r"Low-carbon hegemony", r"High-carbon hegemony"]

    #print(emissions_networks.shape)
    #quit()
    emissions_networks = np.transpose(emissions_networks, (0,2,1,3,4))#transpose to shift
    #print(emissions_networks.shape)
    
    for i, network_em in enumerate(emissions_networks):
        for j, tau_em in enumerate(network_em): 
            for k in range(len(tau_em)):
                Data = tau_em[k]
                #mu_emissions = Data.mean(axis=1)
                mu_emissions, lower_bound, upper_bound = calc_bounds(Data, 0.95)
                if i < 2:
                    axes[j][i].plot(property_vals, mu_emissions, label= homo_title_list[k], c=colors_scenarios[k])
                else:
                    axes[j][i].plot(property_vals, mu_emissions, label= heg_list[k], c=colors_scenarios[k])
                
                # Plot the 95% confidence interval as a shaded area
                axes[j][i].fill_between(property_vals, lower_bound, upper_bound, color=colors_scenarios[k], alpha=0.3)
            axes[j][i].legend()

        axes[j][i].set_xscale('log')
        #axes[j][i].set_ylabel(r"Homophily, $\phi$")
        axes[0][i].set_title(network_titles[i], fontsize="12")
    
    #DEAL WITH y labels
    
    for i, tau_title in enumerate(scenarios_titles):
        axes[i][0].set_ylabel(tau_title)

    fig.supxlabel(r"Confirmation bias, $\theta$", fontsize="12")
    fig.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    #axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    #handles, labels = axes[0][0].get_legend_handles_labels()
    
    #fig.subplots_adjust(right=0.2)  # Adjust right margin to make space for legend

    #fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize="10")

    #plt.tight_layout()
    

    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_phi_xlog_confidence_invert"
    fig.savefig(f + ".png", dpi=600, format="png", bbox_inches='tight')

def plot_emissions_simple_homo_invert(
    fileName, emissions_networks, scenarios_titles, property_vals, colors_scenarios, network_titles
):
    
    fig, axes = plt.subplots(ncols=len(emissions_networks), nrows=len(emissions_networks[0]), figsize=(15, 8), sharex=True)  # Increased width
    
    homo_title_list = [r"No homophily", r"Low homophily", r"High homophily"]
    heg_list = [r"No homophily", r"Low-carbon hegemony", r"High-carbon hegemony"]

    #print(emissions_networks.shape)
    #quit()
    emissions_networks = np.transpose(emissions_networks, (0,2,1,3,4))#transpose to shift
    #print(emissions_networks.shape)
    
    for i, network_em in enumerate(emissions_networks):
        for j, tau_em in enumerate(network_em): 
            for k in range(len(tau_em)):
                Data = tau_em[k]
                mu_emissions = Data.mean(axis=1)
                if i < 2:
                    axes[j][i].plot(property_vals, mu_emissions, label= homo_title_list[k], c=colors_scenarios[k])
                else:
                    axes[j][i].plot(property_vals, mu_emissions, label= heg_list[k], c=colors_scenarios[k])
                data_trans = Data.T
                for v in range(len(data_trans)):
                    axes[j][i].plot(property_vals, data_trans[v], color=colors_scenarios[k], alpha=0.1)
            axes[j][i].legend()

        #axes[j][i].set_xscale('log')
        #axes[j][i].set_ylabel(r"Homophily, $\phi$")
        axes[0][i].set_title(network_titles[i], fontsize="12")
    
    #DEAL WITH y labels
    
    for i, tau_title in enumerate(scenarios_titles):
        axes[i][0].set_ylabel(tau_title)

    fig.supxlabel(r"Confirmation bias, $\theta$", fontsize="12")
    fig.supylabel(r"Cumulative carbon emissions, E", fontsize="12")
    #axes[0].set_ylabel(r"Cumulative carbon emissions, E", fontsize="12")
    #handles, labels = axes[0][0].get_legend_handles_labels()
    
    #fig.subplots_adjust(right=0.2)  # Adjust right margin to make space for legend

    #fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize="10")

    #plt.tight_layout()
    

    plotName = fileName + "/Plots"
    f = plotName + "/network_emissions_simple_conf_invert"
    fig.savefig(f + ".png", dpi=600, format="png", bbox_inches='tight')

def main(
    fileName
    ) -> None: 
    name = "Set2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors_scenarios = cmap.colors  # type: list
    ############################
    #print("base_params", base_params)
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    network_titles = ["Small-World", "Stochastic Block Model", "Scale-Free"]
    emissions_array = load_object(fileName + "/Data", "emissions_array")
    #scenarios_titles = [r"No carbon price, $\tau = 0$", r"Low carbon price, $\tau = 0.1$", r"High carbon price, $\tau = 1$"]
    scenarios_titles = [r"$\tau = 0$", r"$\tau = 0.1$", r"$\tau = 1$"]
    base_params =  load_object(fileName + "/Data", "base_params")
    print(base_params )
    #quit()
    plot_emissions_simple_homo(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    #plot_emissions_simple_xlog_homo_confidence(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    #plot_emissions_simple_xlog_homo_confidence_invert(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    plot_emissions_simple_homo_invert(fileName, emissions_array, scenarios_titles, property_values_list, colors_scenarios, network_titles)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/conf_homo_12_23_16__20_09_2024",
    )