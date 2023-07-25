
# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from package.resources.utility import get_cmap_colours
import numpy as np

def plot_consumption_ratio_time_series(
    fileName: str, Data_list,property_title, property_save, property_vals, seed_reps, time_array
):


    cmap = get_cmap_colours(seed_reps)

    #print(c,emissions_final)
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6), constrained_layout=True, sharex=True, sharey= True)

    for i, ax in enumerate(axes.flat):
        Data = Data_list[i]
        for j in range(seed_reps):
            #print("Data[j]", Data[j])
            ax.plot(time_array, Data[j], color = cmap(j))

        ax.set_title(property_title + " = %s" %(round(property_vals[i], 3))) 

    axes[1][0].set_xlabel("Step")
    axes[1][1].set_xlabel("Step")
    axes[0][0].set_ylabel(r"Carbon Emissions Flow")
    axes[1][0].set_ylabel(r"Carbon Emissions Flow")

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "seperate_time_series"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def main(
    fileName = "results/scenario_comparison_15_47_49__18_07_2023",
    ) -> None: 

    ############################

    emissions_flow_timeseries_array = load_object(fileName + "/Data", "emissions_flow_timeseries_array")
    property_varied = load_object(fileName + "/Data", "property_varied")
    property_varied_title = load_object(fileName + "/Data", "property_varied_title")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    base_params = load_object(fileName + "/Data", "base_params")
    time_array = load_object(fileName + "/Data", "time_array")
   

    plot_consumption_ratio_time_series(fileName, emissions_flow_timeseries_array, property_varied_title, property_varied, property_values_list, base_params["seed_reps"],time_array)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/emisisons_flow_ratio_consumption_10_53_58__25_07_2023"
    )

