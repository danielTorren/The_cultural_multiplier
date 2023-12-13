"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
import numpy as np
from matplotlib.animation import FuncAnimation
from package.resources.utility import ( 
    load_object,
)
import pandas as pd
import seaborn as sns
from package.resources.plot import (
    plot_identity_timeseries,
    plot_value_timeseries,
    plot_attitude_timeseries,
    plot_total_carbon_emissions_timeseries,
    plot_weighting_matrix_convergence_timeseries,
    plot_cultural_range_timeseries,
    plot_average_identity_timeseries,
    plot_joint_cluster_micro,
    print_live_initial_identity_network,
    live_animate_identity_network_weighting_matrix,
    plot_low_carbon_preferences_timeseries,
    plot_total_flow_carbon_emissions_timeseries
)

def plot_Z_timeseries(fileName, Data, dpi_save,latex_bool = False):

    fig, ax = plt.subplots(figsize=(10,6))
    y_title = r"$Z_{t,i}$"

    for v in Data.agent_list:
        ax.plot(np.asarray(Data.history_time), np.asarray(v.history_Z))
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        #ax.set_ylim(0, 1)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_Z_timeseries"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_consumption_no_burn_in(fileName, data, dpi_save):

    y_title = r"Quantity"

    fig, axes = plt.subplots(nrows=2,ncols=2, constrained_layout=True)
    inset_ax = axes[1][0].inset_axes([0.4, 0.4, 0.55, 0.55])

    for v in range(data.N):
        data_indivdiual = data.agent_list[v]
        
        axes[0][0].plot(np.asarray(data.history_time),data_indivdiual.history_H_1)
        axes[0][1].plot(np.asarray(data.history_time),data_indivdiual.history_H_2)
        axes[1][0].plot(np.asarray(data.history_time),data_indivdiual.history_L_1)
        axes[1][1].plot(np.asarray(data.history_time),data_indivdiual.history_L_2)
        inset_ax.plot(data.history_time[0:20],data_indivdiual.history_L_1[0:20])
     # [x, y, width, height], x, y are norm 1
    #print("yo",data.history_time[0:20],data_indivdiual.history_L_1[0:20] )
    
    #inset_ax.set_ylim(0,1)
    axes[1][0].set_ylim(0,0.1)
    axes[1][1].set_ylim(0,0.008)

    axes[0][0].set_title("H_1")
    axes[0][1].set_title("H_2")
    axes[1][0].set_title("L_1")
    axes[1][1].set_title("L_2")


    fig.suptitle(r"sector 1 is the luxury and 2 is the basic good, a = 0.8")
    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/quantity_timeseries_preference_no_burn"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_consumption(fileName, data, dpi_save):

    y_title = r"Quantity"

    fig, axes = plt.subplots(nrows=2,ncols=2, constrained_layout=True)

    for v in range(data.N):
        data_indivdiual = data.agent_list[v]
        
        axes[0][0].plot(np.asarray(data.history_time),data_indivdiual.history_H_1)
        axes[0][1].plot(np.asarray(data.history_time),data_indivdiual.history_H_2)
        axes[1][0].plot(np.asarray(data.history_time),data_indivdiual.history_L_1)
        axes[1][1].plot(np.asarray(data.history_time),data_indivdiual.history_L_2)

    axes[0][0].set_title("H_1")
    axes[0][1].set_title("H_2")
    axes[1][0].set_title("L_1")
    axes[1][1].set_title("L_2")


    fig.suptitle(r"sector 1 is the luxury and 2 is the basic good, a = 0.8")
    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/quantity_timeseries_preference"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_emissions_individuals(fileName, data, dpi_save):

    y_title = r"Individuals' emissions flow"

    fig, ax = plt.subplots(constrained_layout=True)

    for v in range(data.N):
        data_indivdiual = data.agent_list[v]
        
        ax.plot(data.history_time,data_indivdiual.history_flow_carbon_emissions)

    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    plotName = fileName + "/Prints"

    f = plotName + "/indi_emisisons_flow_timeseries_preference"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_low_carbon_adoption_timeseries(fileName, data, emissions_threshold_list, dpi_save):

    y_title = r"Low carbon emissions proportion"

    fig, ax = plt.subplots(constrained_layout=True)

    data_matrix = []
    for v in range(data.N):
        data_matrix.append(data.agent_list[v].history_flow_carbon_emissions)

    N = data.N

    data_matrix_trans = np.asarray(data_matrix).T

    for emissions_threshold  in emissions_threshold_list:
        adoption_time_series = [sum([1 if x < emissions_threshold else 0 for x in j])/N for j in data_matrix_trans]
        #print("adoption_time_series",adoption_time_series)
        ax.plot(data.history_time,adoption_time_series, label = r"$E_{Thres} = $" + str(emissions_threshold))

    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    ax.legend()
    plotName = fileName + "/Prints"

    f = plotName + "/plot_low_carbon_adoption_timeserieds" + str(len(emissions_threshold_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_low_carbon_adoption_transition_timeseries(fileName, data,emissions_threshold_range, dpi_save):

    y_title = r"Low carbon emissions proportion"

    fig, ax = plt.subplots(constrained_layout=True)

    data_matrix = []
    for v in range(data.N):
        data_matrix.append(data.agent_list[v].history_flow_carbon_emissions)

    N = data.N

    data_matrix_trans = np.asarray(data_matrix).T

    lowest_threshold = None
    # Iterate through possible thresholds
    for emissions_threshold in emissions_threshold_range:  # Adjust the range as needed
        # Calculate the proportion of agents above the current threshold at the last timestep
        last_timestep_proportion = np.mean(data_matrix_trans[-1, :] < emissions_threshold)

        # Check if the proportion is 1 (all agents are below the threshold)
        if last_timestep_proportion == 1:
            lowest_threshold = emissions_threshold
            break  # Stop iterating if the condition is met

    # Check if a threshold was found
    if lowest_threshold is not None:
        #print("Lowest emissions threshold:", lowest_threshold)
        
        lower = (lowest_threshold - 0.000001)
        adoption_time_series = [sum([1 if x < lowest_threshold else 0 for x in j])/N for j in data_matrix_trans]
        ax.plot(data.history_time,adoption_time_series, label = r"$E_{Thres} = $" + str(lowest_threshold))
        adoption_time_series = [sum([1 if x < (lower) else 0 for x in j])/N for j in data_matrix_trans]
        ax.plot(data.history_time,adoption_time_series, label = r"$E_{Thres} = $" + str(lower))
    else:
        print("No threshold found that meets the condition.")



    ax.set_ylabel(y_title)
    ax.set_xlabel("Time")
    ax.legend()
    plotName = fileName + "/Prints"

    f = plotName + "/plot_low_carbon_adoption_timeserieds"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")



"""
def pde_value_snapshot(fileName, data_sim,property_plot,, snapsdpi_save):

    time_series = data_sim.history_time
    data_list = []  
    
    for v in range(data_sim.N):
            data_list.append(np.asarray(data_sim.agent_list[v].property_plot))

    data_df = 
    data = pd.DataFrame({
        'Time': time_series,  # Your time points
        'Distribution': data_df,  # Your distribution vectors
    })

    for index, row in data.iterrows():
        sns.kdeplot(row['Distribution'], label=f'Time {row['Time']}')

    # You can add labels and titles to your plot
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Distribution Over Time')

    # Add a legend to distinguish the different time points
plt.legend()

    f = plotName + "/plot_low_carbon_adoption_timeserieds"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")
"""

def variable_animation_distribution(fileName, data_sim,property_plot,x_axis_label, dpi_save):

    time_series = data_sim.history_time
    data_list = []  
    for v in range(data_sim.N):
        data_list.append(np.asarray(eval("data_sim.agent_list[v].%s" % property_plot)))
    
    data_matrix = np.asarray(data_list)
    data_matrix_T = data_matrix.T
    data_list_list = data_matrix_T.tolist()

    # Example data
    data = pd.DataFrame({
        'Time': time_series,
        'Distribution': data_list_list
    })

    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Animation: " + property_plot)
    sns.set_style("whitegrid")

    # Set up the initial KDE plot
    initial_kde = sns.kdeplot(data['Distribution'].iloc[0], color='b', label=f"Time {data['Time'].iloc[0]}")
    ax.set_xlabel('Values')
    ax.set_ylabel('Density')
    ax.set_title(f"Distribution Over Time - Time {data['Time'].iloc[0]}")
    ax.legend(loc='upper right')

    # Define the update function to animate the KDE plot
    def update(frame):
        ax.clear()
        kde = sns.kdeplot(data['Distribution'].iloc[frame], color='b', label=f"Time {data['Time'].iloc[frame]}")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel('Density')
        ax.set_title(f"Distribution Over Time - Time {data['Time'].iloc[frame]}")
        ax.legend(loc='upper right')

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(data), repeat=False)
    return animation

def fixed_animation_distribution(fileName, data_sim,property_plot,x_axis_label, direction, dpi_save,save_bool):

    time_series = data_sim.history_time
    data_list = []  
    for v in range(data_sim.N):
        data_list.append(np.asarray(eval("data_sim.agent_list[v].%s" % property_plot)))
    
    data_matrix = np.asarray(data_list)

    min_lim = np.min(data_matrix)
    max_lim = np.max(data_matrix)

    data_matrix_T = data_matrix.T
    data_list_list = data_matrix_T.tolist()

    # Example data
    data = pd.DataFrame({
        'Time': time_series,
        'Distribution': data_list_list
    })

    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("whitegrid")


    # Set up the initial KDE plot
    if direction == "y":
        initial_kde = sns.kdeplot(y = data['Distribution'].iloc[0], color='b', label=f"Time {data['Time'].iloc[0]}")
        ax.set_ylabel(x_axis_label)
        ax.set_xlabel('Density')
        ax.set_ylim(min_lim,max_lim)
    else:
        initial_kde = sns.kdeplot(x = data['Distribution'].iloc[0], color='b', label=f"Time {data['Time'].iloc[0]}")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel('Density')
        ax.set_xlim(min_lim,max_lim)
    #ax.set_title(f"Distribution Over Time - Time {data['Time'].iloc[0]}")
    ax.legend(loc='upper right')

    # Define the update function to animate the KDE plot
    def update(frame):
        ax.clear()
        if direction == "y":
            kde = sns.kdeplot(y=data['Distribution'].iloc[frame], color='b', label=f"Time {data['Time'].iloc[frame]}/{time_series[-1]}")
            ax.set_ylabel(x_axis_label)
            ax.set_ylim(min_lim,max_lim)
            ax.set_xlabel('Density')
        else:
            kde = sns.kdeplot(x=data['Distribution'].iloc[frame], color='b', label=f"Time {data['Time'].iloc[frame]}/{time_series[-1]}")
            ax.set_xlabel(x_axis_label)
            ax.set_xlim(min_lim,max_lim)
            ax.set_ylabel('Density')
        #ax.set_title(f"Distribution Over Time - Time {data['Time'].iloc[frame]}")
        ax.legend(loc='upper right')

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(data), repeat_delay=100,interval=0.01)
    return animation

def multi_col_fixed_animation_distribution(fileName, data_sim,property_plot,x_axis_label, direction, dpi_save,save_bool):

    time_series = data_sim.history_time
    data_list = []  
    for v in range(data_sim.N):
        data_list.append(np.asarray(eval("data_sim.agent_list[v].%s" % property_plot)))
    
    data_matrix = np.asarray(data_list)#[N, T, M]

    reshaped_array = np.transpose(data_matrix, (2, 1, 0))#[M, T, N] SO COOL THAT YOU CAN SPECIFY REORDING WITH TRANSPOSE!

    min_lim = np.min(data_matrix)
    max_lim = np.max(data_matrix)

    #data_matrix_T = data_matrix.T
    #data_list_list = data_matrix_T.tolist()

    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("whitegrid")

    data_pd_list = []
    for data_matrix_T in reshaped_array:
        # Example data
        data = pd.DataFrame({
            'Time': time_series,
            'Distribution': data_matrix_T.tolist()
        })
        data_pd_list.append(data)

    # Set up the initial KDE plot
    if direction == "y":
        for i,data in enumerate(data_pd_list):
            initial_kde = sns.kdeplot(y = data['Distribution'].iloc[0], label="$\sigma_{%s}$ = %s" % (i+1,data_sim.low_carbon_substitutability_array[i] ))
        ax.set_ylabel(x_axis_label)
        ax.set_xlabel('Density')
        ax.set_ylim(min_lim,max_lim)
    else:
        for i,data in enumerate(data_pd_list):
            initial_kde = sns.kdeplot(x = data['Distribution'].iloc[0], label="$\sigma_{%s}$ = %s" % (i+1,data_sim.low_carbon_substitutability_array[i] ))#color='b'
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel('Density')
        ax.set_xlim(min_lim,max_lim)
    #ax.set_title(f"Distribution Over Time - Time {data['Time'].iloc[0]}")
    ax.legend(loc='upper left')
    ax.set_title(f"Steps: {data['Time'].iloc[0]}/{time_series[-1]}")

    # Define the update function to animate the KDE plot
    def update(frame):
        ax.clear()
        if direction == "y":
            for i,data in enumerate(data_pd_list):
                kde = sns.kdeplot(y=data['Distribution'].iloc[frame], label="$\sigma_{%s}$ = %s" % (i+1,data_sim.low_carbon_substitutability_array[i] ))
            ax.set_ylabel(x_axis_label)
            ax.set_ylim(min_lim,max_lim)
            ax.set_xlabel('Density')
        else:
            for i,data in enumerate(data_pd_list):
                kde = sns.kdeplot(x=data['Distribution'].iloc[frame], label="$\sigma_{%s}$ = %s" % (i+1,data_sim.low_carbon_substitutability_array[i] ))
            ax.set_xlabel(x_axis_label)
            ax.set_xlim(min_lim,max_lim)
            ax.set_ylabel('Density')
        ax.set_title(f"Steps: {data['Time'].iloc[frame]}/{time_series[-1]}")
        #ax.set_title(f"Distribution Over Time - Time {data['Time'].iloc[frame]}")
        ax.legend(loc='upper left')

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(data_pd_list[0]), repeat_delay=100,interval=0.01)

    if save_bool:
        # save the video
        animateName = fileName + "/Animations"
        f = (
            animateName
            + "/live_animate_identity_network_weighting_matrix.mp4"
        )
        # print("f", f)
        writervideo = animation.FFMpegWriter(fps=60)
        animation.save(f, writer=writervideo)
    
    return animation

def plot_chi(
    fileName, 
    data, 
    dpi_save,
    ):

    y_title = "$\chi$"

    fig, axes = plt.subplots(nrows=1,ncols=data.M)

    for v in range(data.N):
        data_indivdiual = np.asarray(data.agent_list[v].history_chi_m)
        
        if data.M == 1:
            #print("data_indivdiual",data_indivdiual)
            #quit()
            axes.plot(
                    np.asarray(data.history_time),
                    data_indivdiual
                )
        else:
            for j in range(data.M):
                #print("HI", len(data.history_time), len(data_indivdiual[:,j]))
                #quit()
                axes[j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )

    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/timeseries_chi"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_omega(
    fileName, 
    data, 
    dpi_save,
    ):

    y_title = "$\Omega$"

    fig, axes = plt.subplots(nrows=1,ncols=data.M)

    for v in range(data.N):
        data_indivdiual = np.asarray(data.agent_list[v].history_omega_m)
        
        if data.M == 1:
            #print("data_indivdiual",data_indivdiual)
            #quit()
            axes.plot(
                    np.asarray(data.history_time),
                    data_indivdiual
                )
        else:
            for j in range(data.M):
                #print("HI", len(data.history_time), len(data_indivdiual[:,j]))
                #quit()
                axes[j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )
                #axes[j].set_ylim(0,100)

    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/timeseries_omega"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_consum_ratio(
    fileName, 
    data, 
    dpi_save,
    ):

    y_title = "Consumption ratio, $C_{t,i,m}$"

    fig, axes = plt.subplots(nrows=1,ncols=data.M)

    for v in range(data.N):
        data_indivdiual = np.asarray(data.agent_list[v].history_omega_m)
        
        if data.M == 1:
            #print("data_indivdiual",data_indivdiual)
            #quit()
            data_ind = np.asarray(data_indivdiual)
            consum_ratio = data_ind/(1+data_ind) 
            axes.plot(
                    np.asarray(data.history_time),
                    consum_ratio
                )
            axes.set_ylim(0,1)
        else:
            for j in range(data.M):
                #print("HI", len(data.history_time), len(data_indivdiual[:,j]))
                #quit()
                data_ind = np.asarray(data_indivdiual[:,j])
                consum_ratio = data_ind/(1+data_ind) 
                axes[j].plot(
                    np.asarray(data.history_time),
                    consum_ratio
                )
                axes[j].set_ylim(0,1)

    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/timeseries_consum_ratio"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_L(
    fileName, 
    data, 
    dpi_save,
    ):

    y_title = "L"

    fig, axes = plt.subplots(nrows=1,ncols=data.M)

    for v in range(data.N):
        data_indivdiual = np.asarray(data.agent_list[v].history_L_m)
        
        if data.M == 1:
            #print("data_indivdiual",data_indivdiual)
            #quit()
            axes.plot(
                    np.asarray(data.history_time),
                    data_indivdiual
                )
        else:
            for j in range(data.M):
                #print("HI", len(data.history_time), len(data_indivdiual[:,j]))
                #quit()
                axes[j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )

    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/timeseries_L"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_H(
    fileName, 
    data, 
    dpi_save,
    ):

    y_title = "H"

    fig, axes = plt.subplots(nrows=1,ncols=data.M)

    for v in range(data.N):
        data_indivdiual = np.asarray(data.agent_list[v].history_H_m)
        
        if data.M == 1:
            #print("data_indivdiual",data_indivdiual)
            #quit()
            axes.plot(
                    np.asarray(data.history_time),
                    data_indivdiual
                )
        else:
            for j in range(data.M):
                #print("HI", len(data.history_time), len(data_indivdiual[:,j]))
                #quit()
                axes[j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )

    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/timeseries_H"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    dpi_save = 600,
    ) -> None: 

    cmap_multi = get_cmap("plasma")
    cmap_weighting = get_cmap("Reds")

    norm_zero_one = Normalize(vmin=0, vmax=1)
    cmap = LinearSegmentedColormap.from_list(
        "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
    )

    Data = load_object(fileName + "/Data", "social_network")

    anim_save_bool = False#Need to install the saving thing
    ###PLOTS
    """
    if Data.burn_in_duration == 0:
        plot_consumption_no_burn_in(fileName, Data, dpi_save)
    else:
        plot_consumption(fileName, Data, dpi_save)
    """
    plot_low_carbon_preferences_timeseries(fileName, Data, dpi_save)
    plot_emissions_individuals(fileName, Data, dpi_save)
    plot_identity_timeseries(fileName, Data, dpi_save)
    plot_total_carbon_emissions_timeseries(fileName, Data, dpi_save)
    plot_total_flow_carbon_emissions_timeseries(fileName, Data, dpi_save)
    plot_chi(fileName, Data, dpi_save)
    plot_omega(fileName, Data, dpi_save)
    plot_consum_ratio(fileName, Data, dpi_save)
    plot_L(fileName, Data, dpi_save)
    plot_H(fileName, Data, dpi_save)
    plot_Z_timeseries(fileName, Data, dpi_save)
    #threshold_list = [0.0001,0.0002,0.0005,0.001,0.002,0.003,0.004]
    #emissions_threshold_range = np.arange(0,0.005,0.000001)
    #plot_low_carbon_adoption_timeseries(fileName, Data,threshold_list, dpi_save)
    #plot_low_carbon_adoption_transition_timeseries(fileName, Data,emissions_threshold_range, dpi_save)
    
    #anim_1 = variable_animation_distribution(fileName, Data, "history_identity","Identity", dpi_save,anim_save_bool)
    #anim_2 = varaible_animation_distribution(fileName, Data, "history_flow_carbon_emissions","Individual emissions" , dpi_save,anim_save_bool)

    #anim_3 = fixed_animation_distribution(fileName, Data, "history_identity","Identity","y", dpi_save,anim_save_bool)
    #anim_4 = fixed_animation_distribution(fileName, Data, "history_flow_carbon_emissions","Individual emissions","y",dpi_save,anim_save_bool)
    #anim_5 = fixed_animation_distribution(fileName, Data, "history_utility","Utility","y",dpi_save,anim_save_bool)
    
    #VECTORS
    #anim_4 = multi_col_fixed_animation_distribution(fileName, Data, "history_low_carbon_preferences","Low carbon Preferences","y", dpi_save,anim_save_bool)


    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/single_experiment_16_16_09__04_12_2023",#
    )


