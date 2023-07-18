
# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from package.resources.utility import get_cmap_colours

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

    data_holder = load_object(fileName + "/Data", "data_holder")
    data_holder_consumption_based = load_object(fileName + "/Data", "data_holder_consumption_based")
    property_varied = load_object(fileName + "/Data", "property_varied")
    property_varied_title = load_object(fileName + "/Data", "property_varied_title")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    base_params = load_object(fileName + "/Data", "base_params")
    scenarios= load_object(fileName + "/Data", "scenarios")


    plot_end_points_emissions_scenarios(fileName, data_holder, property_varied_title, property_varied, property_values_list,scenarios,"Attitude learning","attitude")
    plot_end_points_emissions_scenarios(fileName, data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios, "Consumption learning","consumption")

    scatter_end_points_emissions_scenarios(fileName, data_holder, property_varied_title, property_varied, property_values_list,scenarios,"Attitude learning", base_params["seed_reps"],"attitude")
    scatter_end_points_emissions_scenarios(fileName, data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios, "Consumption learning", base_params["seed_reps"],"consumption")

    plot_seperate_end_points_emissions_scenarios(fileName, data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios,"Consumption learning", base_params["seed_reps"],"consumption" )
    plot_seperate_end_points_emissions_scenarios(fileName, data_holder, property_varied_title, property_varied, property_values_list,scenarios,"Attitude learning", base_params["seed_reps"], "attitude")

    plot_consumption_impact(fileName,data_holder,  data_holder_consumption_based, property_varied_title, property_varied, property_values_list,scenarios, base_params["seed_reps"])
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/scenario_comparison_20_14_41__18_07_2023"
    )

