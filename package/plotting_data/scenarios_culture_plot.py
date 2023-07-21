
# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from package.resources.utility import get_cmap_colours
import numpy as np

def plot_consumption_impact(
    fileName: str, Data_holder, Data_holder_consumption_based,property_title, property_save, property_vals, scenarios, seed_reps
):

    cmap = get_cmap_colours(seed_reps)

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6), constrained_layout=True)

    for i, ax in enumerate(axes.flat):
        Data_list = Data_holder[i] - Data_holder_consumption_based[i]
        print("asfdasd,",Data_list)
        mu_emissions =  Data_list.mean(axis=1)
        min_emissions =  Data_list.min(axis=1)
        max_emissions=  Data_list.max(axis=1)

        ax.plot(property_vals, mu_emissions, color = cmap(i))
        ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5, facecolor = cmap(i))

        ax.set_xlabel(scenarios[i])
        ax.set_ylabel(r"Carbon Emissions change ")

    
    fig.suptitle("Emissiosn change from attitude- to consumption-based learning")
        #ax.legend()

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "subrtaction_seperate_scenarios_emissions"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def scatter_seperate_end_points_emissions_scenarios(    
fileName: str, Data_holder, property_title, property_save, property_vals, scenarios,title, seed_reps, learn_type
):

    cmap = get_cmap_colours(seed_reps)

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6), constrained_layout=True)

    for i, ax in enumerate(axes.flat):
        Data_list = Data_holder[i].T #take the transpose so i can plot more easily
        #quit()
        ##print("Data_listData_list",Data_list.shape)
        for j in range(seed_reps):
            ax.scatter(property_vals, Data_list[j], c = cmap(j))

        #Data_list = Data_holder[i]
        #mu_emissions =  Data_list.mean(axis=1)
        #min_emissions =  Data_list.min(axis=1)
        #max_emissions=  Data_list.max(axis=1)

        #ax.plot(property_vals, mu_emissions, color = cmap(i))
        #ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5, facecolor = cmap(i))

        ax.set_xlabel(scenarios[i])
        ax.set_ylabel(r"Carbon Emissions")

        ax.set_title(title)
        #ax.legend()

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "scatter_seperate_scenarios_emissions_" + learn_type
    fig.savefig(f+ ".png", dpi=600, format="png")

def norm_scatter_seperate_end_points_emissions_scenarios(
    fileName: str, Data_holder, property_title, property_save, property_vals, scenarios,title, seed_reps, learn_type
):

    cmap = get_cmap_colours(seed_reps)

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6), constrained_layout=True, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flat):
        Data_list = Data_holder[i]
        first_row = Data_list[0]
        result_matrix = Data_list/first_row#divide the emissions through by the case of a zero carbon price so that its soley the effect of carbon price isolated

        result_matrix_t = result_matrix.T
        for j in range(seed_reps):
            ax.scatter(property_vals, result_matrix_t[j], c = cmap(j))

        ax.set_xlabel("Carbon price")
        ax.set_ylabel(r"Carbon Emissions")
        ax.set_title( scenarios[i])
    fig.suptitle("Normalized " + title)

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "norm_scatter_seperate_scenarios_emissions_" + learn_type
    fig.savefig(f+ ".png", dpi=600, format="png")

def norm_plot_seperate_end_points_emissions_scenarios(
    fileName: str, Data_holder, property_title, property_save, property_vals, scenarios,title, seed_reps, learn_type
):

    cmap = get_cmap_colours(seed_reps)

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6), constrained_layout=True, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flat):
        Data_list = Data_holder[i]
        first_row = Data_list[0]
        result_matrix = Data_list / first_row

        mu_emissions =  result_matrix.mean(axis=1)
        min_emissions =  result_matrix.min(axis=1)
        max_emissions=  result_matrix.max(axis=1)

        ax.plot(property_vals, mu_emissions, color = cmap(i))
        ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5, facecolor = cmap(i))

        ax.set_xlabel("Carbon price")
        ax.set_ylabel(r"Carbon Emissions")
        ax.set_title( scenarios[i])
    fig.suptitle("Normalized " + title)
    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "norm_seperate_scenarios_emissions_" + learn_type
    fig.savefig(f+ ".png", dpi=600, format="png") 


def init_norm_plot_seperate_end_points_emissions_scenarios(    
        fileName: str, Data_holder, init_Data_holder, property_title, property_save, property_vals, scenarios,title, seed_reps, learn_type, steps
):

    cmap = get_cmap_colours(seed_reps)

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6), constrained_layout=True)

    for i, ax in enumerate(axes.flat):
        Data_list = Data_holder[i]/((steps)*init_Data_holder[i])#divide through by the reference emissions

        # if i == 0:
        #    print("this should be all ones:",Data_list[0])
            #print("test -1", Data_holder[i]/((steps-1)*init_Data_holder[i]))# yeah so its defo +1

        mu_emissions =  Data_list.mean(axis=1)
        min_emissions =  Data_list.min(axis=1)
        max_emissions=  Data_list.max(axis=1)

        ax.plot(property_vals, mu_emissions, color = cmap(i))
        ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5, facecolor = cmap(i))

        ax.set_xlabel(scenarios[i])
        ax.set_ylabel(r"Change in Carbon Emissions")

        ax.set_title(title)
        #ax.legend()

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "init_norm_seperate_scenarios_emissions_" + learn_type
    fig.savefig(f+ ".png", dpi=600, format="png")

def plot_seperate_end_points_emissions_scenarios(
    fileName: str, Data_holder, property_title, property_save, property_vals, scenarios,title, seed_reps, learn_type
):

    cmap = get_cmap_colours(seed_reps)

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6), constrained_layout=True)

    for i, ax in enumerate(axes.flat):
        Data_list = Data_holder[i]
        mu_emissions =  Data_list.mean(axis=1)
        min_emissions =  Data_list.min(axis=1)
        max_emissions=  Data_list.max(axis=1)

        ax.plot(property_vals, mu_emissions, color = cmap(i))
        ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5, facecolor = cmap(i))

        ax.set_xlabel(scenarios[i])
        ax.set_ylabel(r"Carbon Emissions")

        ax.set_title(title)
        #ax.legend()

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "seperate_scenarios_emissions_" + learn_type
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_end_points_emissions_scenarios_joint_four(
    fileName: str, Data_set, property_title, property_save, property_vals, scenarios,titles, labels
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,6), sharey="row", constrained_layout=True)

    axes[0][0].grid()
    axes[0][1].grid()
    axes[1][0].grid()
    axes[1][1].grid()

    for j, ax in enumerate(axes.flat):
        for i, scenario  in enumerate(scenarios):
            Data_list = Data_set[j][i]
            mu_emissions =  Data_list.mean(axis=1)
            min_emissions =  Data_list.min(axis=1)
            max_emissions=  Data_list.max(axis=1)

            ax.plot(property_vals, mu_emissions, label = labels[i])
            ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)
            ax.legend()
            ax.set_title(titles[j])

    axes[1][0].set_xlabel(property_title)
    axes[1][1].set_xlabel(property_title)
    axes[0][0].set_ylabel(r"Carbon Emissions stock")
    axes[1][0].set_ylabel(r"Carbon Emissions stock")    

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "scenarios_emissions_joint_four" 
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_social_versus_cultural_multiplier(
    fileName: str, Data_set, property_title, property_save, property_vals, scenarios,titles, labels
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,6), sharey="row", constrained_layout=True)

    axes[0][0].grid()
    axes[0][1].grid()
    axes[1][0].grid()
    axes[1][1].grid()

    for j, ax in enumerate(axes.flat):
        Data_social = Data_set[j][2]
        Data_cultural = Data_set[j][3]

        Data_list = Data_cultural/Data_social

        mu_emissions =  Data_list.mean(axis=1)
        min_emissions =  Data_list.min(axis=1)
        max_emissions=  Data_list.max(axis=1)

        ax.plot(property_vals, mu_emissions)
        ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)
        ax.set_title(titles[j])

    axes[1][0].set_xlabel(property_title)
    axes[1][1].set_xlabel(property_title)
    axes[0][0].set_ylabel(r"Carbon Emissions stock ratio")
    axes[1][0].set_ylabel(r"Carbon Emissions stock ratio")  

    fig.suptitle("Emissions ratio between cultural and social multiplier")  

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "scenarios_social_versus_cultural_multiplier_four" 
    fig.savefig(f+ ".png", dpi=600, format="png") 



def plot_end_points_emissions_scenarios_joint(
    fileName: str, Data_holder,Data_holder_consum, property_title, property_save, property_vals, scenarios,title, title_consum, labels
):

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6), sharey=True, constrained_layout=True)

    axes[0].grid()
    axes[1].grid()

    for i, scenario  in enumerate(scenarios):
        Data_list = Data_holder[i]
        mu_emissions =  Data_list.mean(axis=1)
        min_emissions =  Data_list.min(axis=1)
        max_emissions=  Data_list.max(axis=1)

        axes[0].plot(property_vals, mu_emissions, label = labels[i])
        axes[0].fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)

    axes[0].set_xlabel(property_title)
    axes[0].set_ylabel(r"Carbon Emissions stock")

    axes[0].set_title(title)
    axes[0].legend()
    
    
    for i, scenario  in enumerate(scenarios):
        Data_list = Data_holder_consum[i]
        mu_emissions =  Data_list.mean(axis=1)
        min_emissions =  Data_list.min(axis=1)
        max_emissions=  Data_list.max(axis=1)

        axes[1].plot(property_vals, mu_emissions, label = labels[i])
        axes[1].fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)

    axes[1].set_xlabel(property_title)
    axes[1].set_title(title_consum)
    axes[1].legend()
    

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "scenarios_emissions_joint" 
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_end_points_emissions_scenarios(
    fileName: str, Data_holder, property_title, property_save, property_vals, scenarios,title, learn_type
):

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6))

    for i, scenario  in enumerate(scenarios):
        Data_list = Data_holder[i]
        mu_emissions =  Data_list.mean(axis=1)
        min_emissions =  Data_list.min(axis=1)
        max_emissions=  Data_list.max(axis=1)

        ax.plot(property_vals, mu_emissions, label = scenario)
        ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)

    ax.set_xlabel(property_title)
    ax.set_ylabel(r"Carbon Emissions")

    ax.set_title(title)
    ax.legend()

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "scenarios_emissions_" + learn_type
    fig.savefig(f+ ".png", dpi=600, format="png") 

def scatter_end_points_emissions_scenarios(
    fileName: str, Data_holder, property_title, property_save, property_vals, scenarios,title, seed_reps, learn_type
):


    cmap = get_cmap_colours(seed_reps)

    #print(c,emissions_final)
    fig, ax = plt.subplots(figsize=(10,6))

    for i, scenario  in enumerate(scenarios):
        Data_list = Data_holder[i].T #take the transpose so i can plot more easily
        #quit()
        ##print("Data_listData_list",Data_list.shape)
        for j in range(seed_reps):

            ax.scatter(property_vals, Data_list[j], label = scenario, c = cmap(j))

    ax.set_xlabel(property_title)
    ax.set_ylabel(r"Carbon Emissions")

    ax.set_title(title)

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    unique_handles = list(by_label.values())
    unique_labels = list(by_label.keys())

    ax.legend(unique_handles, unique_labels)

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "scatter_scenarios_emissions_" + learn_type
    fig.savefig(f+ ".png", dpi=600, format="png") 

def main(
    fileName = "results/scenario_comparison_15_47_49__18_07_2023",
    ) -> None: 

    ############################

    data_holder_attitude_learn_attitude_identity = load_object(fileName + "/Data", "data_holder_attitude_learn_attitude_identity")
    data_holder_attitude_learn_consumption_identity = load_object( fileName + "/Data", "data_holder_attitude_learn_consumption_identity")
    data_holder_consumption_learn_attitude_identity = load_object( fileName + "/Data", "data_holder_consumption_learn_attitude_identity")
    data_holder_consumption_learn_consumption_identity = load_object( fileName + "/Data", "data_holder_consumption_learn_consumption_identity")

    data_set = [data_holder_attitude_learn_attitude_identity,data_holder_attitude_learn_consumption_identity,data_holder_consumption_learn_attitude_identity,data_holder_consumption_learn_consumption_identity]

    titles = ["Attitude learning, Attitude identity", "Attitude learning, Consumption identity", "Consumption learning, Attitude identity","Consumption learning, Consumption identity" ]

    property_varied = load_object(fileName + "/Data", "property_varied")
    property_varied_title = load_object(fileName + "/Data", "property_varied_title")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    base_params = load_object(fileName + "/Data", "base_params")
    scenarios= load_object(fileName + "/Data", "scenarios")


    labels = ["Static preferences", "Static weightings","Social multiplier", "Cultural multiplier"]
    #plot_end_points_emissions_scenarios(fileName, data_holder, property_varied_title, property_varied, property_values_list,scenarios,"Attitude learning","attitude")
    #plot_end_points_emissions_scenarios(fileName, data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios, "Consumption learning","consumption")
    
    #plot_end_points_emissions_scenarios_joint(fileName, data_holder, data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios, "Attitude learning", "Consumption learning",labels)

    plot_end_points_emissions_scenarios_joint_four(
        fileName,  data_set, property_varied_title, property_varied, property_values_list,scenarios,titles, labels
        )
    
    plot_social_versus_cultural_multiplier(
        fileName,  data_set, property_varied_title, property_varied, property_values_list,scenarios,titles, labels
    )

    #scatter_end_points_emissions_scenarios(fileName, data_holder, property_varied_title, property_varied, property_values_list,scenarios,"Attitude learning", base_params["seed_reps"],"attitude")
    #scatter_end_points_emissions_scenarios(fileName, data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios, "Consumption learning", base_params["seed_reps"],"consumption")

    #scatter_seperate_end_points_emissions_scenarios(fileName, data_holder, property_varied_title, property_varied, property_values_list,scenarios,"Attitude learning", base_params["seed_reps"],"attitude")
    #scatter_seperate_end_points_emissions_scenarios(fileName, data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios, "Consumption learning", base_params["seed_reps"],"consumption")

    #plot_seperate_end_points_emissions_scenarios(fileName, data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios,"Consumption learning", base_params["seed_reps"],"consumption" )
    #plot_seperate_end_points_emissions_scenarios(fileName, data_holder, property_varied_title, property_varied, property_values_list,scenarios,"Attitude learning", base_params["seed_reps"], "attitude")

    #plot_consumption_impact(fileName,data_holder,  data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios, base_params["seed_reps"])
    
    #norm_plot_seperate_end_points_emissions_scenarios(fileName, data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios,"Consumption learning", base_params["seed_reps"],"consumption" )
    #norm_plot_seperate_end_points_emissions_scenarios(fileName, data_holder, property_varied_title, property_varied, property_values_list,scenarios,"Attitude learning", base_params["seed_reps"], "attitude")

    #norm_scatter_seperate_end_points_emissions_scenarios(fileName, data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios,"Consumption learning", base_params["seed_reps"],"consumption" )
    #norm_scatter_seperate_end_points_emissions_scenarios(fileName, data_holder, property_varied_title, property_varied, property_values_list,scenarios,"Attitude learning", base_params["seed_reps"], "attitude")

    #the +1 in the steps is to account for the 0th step i think, The emissions should be the same for the case of 0 carbon price and fixed preferecens
    #init_norm_plot_seperate_end_points_emissions_scenarios(fileName, data_holder_consumption_based,init_data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios,"Consumption learning", base_params["seed_reps"],"consumption" , base_params["time_steps_max"]+1)
    #init_norm_plot_seperate_end_points_emissions_scenarios(fileName, data_holder, init_data_holder, property_varied_title, property_varied, property_values_list,scenarios,"Attitude learning", base_params["seed_reps"], "attitude", base_params["time_steps_max"] +1)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/scenario_comparison_18_18_54__20_07_2023"
    )

