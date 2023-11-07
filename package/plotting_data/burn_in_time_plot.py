
# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from package.resources.utility import get_cmap_colours
import numpy as np

def plot_emissions_flow_time_series(
    fileName: str, Data_list,property_title, property_save, property_vals, seed_reps, time_array
):


    cmap = get_cmap_colours(seed_reps)

    #print(c,emissions_final)
    rows_cols = int((len(property_vals))**0.5)
    fig, axes = plt.subplots(nrows=rows_cols, ncols=rows_cols,figsize=(10,6), constrained_layout=True, sharex=True, sharey= True)

    for i, ax in enumerate(axes.flat):
        Data = Data_list[i]
        for j in range(seed_reps):
            #print("Data[j]", Data[j])
            ax.plot(time_array, Data[j], color = cmap(j))

        ax.set_title(property_title + " = %s" %(round(property_vals[i], 3))) 

    fig.supxlabel("Step")
    fig.supylabel(r"Carbon emissions flow")

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "emissions_flow_time_series"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_emissions_stock_time_series(
    fileName: str, Data_list,property_title, property_save, property_vals, seed_reps, time_array
):

    cmap = get_cmap_colours(seed_reps)

    rows_cols = int((len(property_vals))**0.5)
    fig, axes = plt.subplots(nrows=rows_cols, ncols=rows_cols,figsize=(10,6), constrained_layout=True, sharex=True, sharey= True)

    for i, ax in enumerate(axes.flat):
        Data = Data_list[i]
        for j in range(seed_reps):
            #print("Data[j]", Data[j])
            ax.plot(time_array, Data[j], color = cmap(j))

        ax.set_title(property_title + " = %s" %(round(property_vals[i], 3))) 

    fig.supxlabel("Step")
    fig.supylabel(r"Carbon emissions flow")

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "emissions_stock_time_series"
    fig.savefig(f+ ".png", dpi=600, format="png") 

def plot_final_emissions_stock(
    fileName: str, Data_array,property_title, property_save, property_vals, seed_reps, time_array
):
    
    data_wierd = Data_array[:, :, -1:]
    desired_shape = data_wierd.shape
    #print("desired_shape", desired_shape, desired_shape[0])
    data = data_wierd.reshape(desired_shape[0], desired_shape[1])
    #print("data",data)

    #print(c,emissions_final)
    
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)

    mu_emissions =  data.mean(axis=1)
    min_emissions =  data.min(axis=1)
    max_emissions=  data.max(axis=1)

    ax.plot(property_vals, mu_emissions)
    ax.fill_between(property_vals, min_emissions, max_emissions, alpha=0.5)
    
    ax.set_xlabel(property_title)
    ax.set_ylabel(r"Carbon Emissions stock") 

    #print("what worong")
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_save + "emissions_stock" 
    fig.savefig(f+ ".png", dpi=600, format="png")

def main(
    fileName = "results/scenario_comparison_15_47_49__18_07_2023",
    ) -> None: 

    ############################

    emissions_flow_timeseries_array = load_object(fileName + "/Data", "emissions_flow_timeseries_array")
    emissions_stock_timeseries_array = load_object(fileName + "/Data", "emissions_stock_timeseries_array")
    property_varied = load_object(fileName + "/Data", "property_varied")
    property_varied_title ="$T_{B}$" #load_object(fileName + "/Data", "property_varied_title")
    property_values_list = load_object(fileName + "/Data", "property_values_list")
    base_params = load_object(fileName + "/Data", "base_params")
    time_array = load_object(fileName + "/Data", "time_array")
   
    ################# CAN ONLY HAVE 4 REPS!!!
    #print("len(property_values_list)", len(property_values_list), (len(property_values_list)**0.5))
    len_res = (len(property_values_list)**0.5)
    #print("len_res == int(len_res)", len_res == int(len_res))
    if len_res == int(len_res):
        plot_emissions_flow_time_series(fileName, emissions_flow_timeseries_array, property_varied_title, property_varied, property_values_list, base_params["seed_reps"],time_array)
        plot_emissions_stock_time_series(fileName, emissions_stock_timeseries_array, property_varied_title, property_varied, property_values_list, base_params["seed_reps"],time_array)
    
    plot_final_emissions_stock(fileName, emissions_stock_timeseries_array, property_varied_title, property_varied, property_values_list, base_params["seed_reps"],time_array)
    #print("emissions_stock_timeseries_array :",emissions_stock_timeseries_array, emissions_stock_timeseries_array.shape)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName="results/burn_in_duration_sweep_09_26_53__07_11_2023"
    )

