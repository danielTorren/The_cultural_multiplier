"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap, ScalarMappable
import numpy as np
import matplotlib.markers as mmarkers
from matplotlib.animation import FuncAnimation
from package.resources.utility import ( 
    load_object,
)
import pandas as pd
import seaborn as sns
import networkx as nx
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
    plot_total_flow_carbon_emissions_timeseries,
    prod_pos,
    plot_SBM_low_carbon_preferences_timeseries
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

def plot_network_end_preferences(    
        fileName, 
        data, 
        cmap,
        dpi_save,
        node_sizes,
        norm
    ):

    fig, axes = plt.subplots(nrows=1,ncols=data.M,figsize=(10,6))

    

    data_list = []
    for v in range(data.N):
        data_list.append(data.agent_list[v].low_carbon_preferences)
    
    data_array = np.asarray(data_list).T#now its M by N
    
    G = data.network

    if data.network_type == "SW":
        pos = prod_pos("circular", G)
    else:
        pos = nx.spring_layout(G,seed=1)  # You can use other layout algorithms

    if data.M == 1:
        colour_adjust = norm(data_array)
        ani_step_colours = cmap(colour_adjust)
        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=axes,
            pos=pos,
            node_size=node_sizes,
            edgecolors="black",
        )
    else:
        for i, ax in enumerate(axes.flat):
            colour_adjust = norm(data_array[i])
            ani_step_colours = cmap(colour_adjust)
            nx.draw(
                G,
                node_color=ani_step_colours,
                ax=ax,
                pos=pos,
                node_size=node_sizes,
                edgecolors="black",
            )
            ax.set_title("Sector = %s" % (i+1))
    
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[-1]
    )
    cbar.set_label(r"Final preference, $A_{t_{max},i,m}$")

    fig.suptitle("Final preference")

    plotName = fileName + "/Prints"

    f = plotName + "/network_end_preference"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_network_start_preferences(    
        fileName, 
        data, 
        cmap,
        dpi_save,
        node_sizes,
        norm
    ):

    fig, axes = plt.subplots(nrows=1,ncols=data.M,figsize=(10,6))

    data_list = []
    for v in range(data.N):
        data_list.append(data.agent_list[v].history_low_carbon_preferences[0])
    
    data_array = np.asarray(data_list).T#now its M by N
    
    G = data.network

    if data.network_type == "SW":
        pos = prod_pos("circular", G)
    else:
        pos = nx.spring_layout(G,seed=1)  # You can use other layout algorithms

    if data.M == 1:
        colour_adjust = norm(data_array)
        ani_step_colours = cmap(colour_adjust)
        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=axes,
            pos=pos,
            node_size=node_sizes,
            edgecolors="black",
        )
    else:
        for i, ax in enumerate(axes.flat):
            colour_adjust = norm(data_array[i])
            ani_step_colours = cmap(colour_adjust)
            nx.draw(
                G,
                node_color=ani_step_colours,
                ax=ax,
                pos=pos,
                node_size=node_sizes,
                edgecolors="black",
            )
            ax.set_title("Sector = %s" % (i+1))
    
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[-1]
    )
    cbar.set_label(r"Initial preference, $A_{0,i,m}$")

    fig.suptitle("Initial preference")

    plotName = fileName + "/Prints"

    f = plotName + "/network_start_preference"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_SBM_network_end_preferences(    
        fileName, 
        data, 
        cmap,
        dpi_save,
        node_sizes,
        norm,
        block_markers_list,
        legend_loc,
        lines_alpha
    ):

    fig, axes = plt.subplots(nrows=1,ncols=data.M,figsize=(10,6))

    data_list = []
    for v in range(data.N):
        data_list.append(data.agent_list[v].low_carbon_preferences)
    
    data_array = np.asarray(data_list).T#now its M by N

    if data.SBM_block_num > len(block_markers_list):
        raise ValueError("Not enough markers for number of blocks")
    
    #node_shapes = np.asarray([[block_markers_list[i]]*data.SBM_block_sizes[i] for i in range(data.SBM_block_num)]).flatten()
    present_block_markers_list = [block_markers_list[i] for i in range(data.SBM_block_num)]
    node_shapes = np.repeat(present_block_markers_list,  data.SBM_block_sizes, axis=0)  

    G = data.network

    if data.network_type == "SW":
        pos = prod_pos("circular", G)
    else:
        pos = nx.spring_layout(G, seed=1)  # You can use other layout algorithms

    if data.M == 1:
        colour_adjust = norm(data_array)
        ani_step_colours = cmap(colour_adjust)

        for j,node in enumerate(G.nodes()):
            G.nodes[node]['color'] = ani_step_colours[j]
            G.nodes[node]['shape'] = node_shapes[j]

        # Draw the nodes for each shape with the shape specified
        for shape in set(node_shapes):
            # the nodes with the desired shapes
            node_list = [node for node in G.nodes() if G.nodes[node]['shape'] == shape]
            nx.draw_networkx_nodes(
                G,
                pos,
                ax = ax,
                nodelist = node_list,
                node_size = node_sizes,
                node_color= [G.nodes[node]['color'] for node in node_list],
                node_shape = shape
            )
            nx.draw_networkx_edges(G,pos, ax=ax, alpha=lines_alpha) # draw edges

    else:  
        for i, ax in enumerate(axes.flat):

            colour_adjust = norm(data_array[i])
            ani_step_colours = cmap(colour_adjust)

            for j,node in enumerate(G.nodes()):
                G.nodes[node]['color'] = ani_step_colours[j]
                G.nodes[node]['shape'] = node_shapes[j]
    
            # Draw the nodes for each shape with the shape specified
            for shape in set(node_shapes):
                # the nodes with the desired shapes
                node_list = [node for node in G.nodes() if G.nodes[node]['shape'] == shape]
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax = ax,
                    nodelist = node_list,
                    node_size = node_sizes,
                    node_color= [G.nodes[node]['color'] for node in node_list],
                    node_shape = shape
                )
                nx.draw_networkx_edges(G,pos, ax=ax, alpha=lines_alpha) # draw edges

            ax.set_title("Sector = %s" % (i+1))

    # Add legend
    unique_shapes = list(set([G.nodes[node]['shape'] for node in G.nodes()]))
    legend_labels = [f"Block {i+1}" for i in range(len(unique_shapes))]
    for i,marker in enumerate(unique_shapes):
        axes[-1].scatter([],[], label=legend_labels[i], marker = marker, c="black")
    axes[-1].legend(loc=legend_loc)
    
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes.ravel().tolist()
    )
    cbar.set_label(r"Final preference, $A_{t_{max},i,m}$")

    fig.suptitle("Final preference")

    plotName = fileName + "/Prints"

    f = plotName + "/SBM_network_end_preference"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_SBM_network_start_preferences(    
        fileName, 
        data, 
        cmap,
        dpi_save,
        node_sizes,
        norm,
        block_markers_list,
        legend_loc,
        lines_alpha
    ):

    fig, axes = plt.subplots(nrows=1,ncols=data.M,figsize=(10,6))

    data_list = []
    for v in range(data.N):
        data_list.append(data.agent_list[v].history_low_carbon_preferences[0])
    
    data_array = np.asarray(data_list).T#now its M by N

    if data.SBM_block_num > len(block_markers_list):
        raise ValueError("Not enough markers for number of blocks")
    
    #node_shapes = np.asarray([[block_markers_list[i]]*data.SBM_block_sizes[i] for i in range(data.SBM_block_num)]).flatten()
    present_block_markers_list = [block_markers_list[i] for i in range(data.SBM_block_num)]
    node_shapes = np.repeat(present_block_markers_list,  data.SBM_block_sizes, axis=0)   

    G = data.network

    if data.network_type == "SW":
        pos = prod_pos("circular", G)
    else:
        pos = nx.spring_layout(G, seed=1)  # You can use other layout algorithms

    if data.M == 1:
        colour_adjust = norm(data_array)
        ani_step_colours = cmap(colour_adjust)

        for j,node in enumerate(G.nodes()):
            G.nodes[node]['color'] = ani_step_colours[j]
            G.nodes[node]['shape'] = node_shapes[j]

        # Draw the nodes for each shape with the shape specified
        for shape in set(node_shapes):
            # the nodes with the desired shapes
            node_list = [node for node in G.nodes() if G.nodes[node]['shape'] == shape]
            nx.draw_networkx_nodes(
                G,
                pos,
                ax = ax,
                nodelist = node_list,
                node_size = node_sizes,
                node_color= [G.nodes[node]['color'] for node in node_list],
                node_shape = shape
            )
            nx.draw_networkx_edges(G,pos, ax=ax, alpha=lines_alpha) # draw edges

    else:  
        for i, ax in enumerate(axes.flat):

            colour_adjust = norm(data_array[i])
            ani_step_colours = cmap(colour_adjust)

            for j,node in enumerate(G.nodes()):
                G.nodes[node]['color'] = ani_step_colours[j]
                G.nodes[node]['shape'] = node_shapes[j]
    
            # Draw the nodes for each shape with the shape specified
            for shape in set(node_shapes):
                # the nodes with the desired shapes
                node_list = [node for node in G.nodes() if G.nodes[node]['shape'] == shape]
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax = ax,
                    nodelist = node_list,
                    node_size = node_sizes,
                    node_color= [G.nodes[node]['color'] for node in node_list],
                    node_shape = shape
                )
                nx.draw_networkx_edges(G,pos, ax=ax, alpha=lines_alpha) # draw edges

            ax.set_title("Sector = %s" % (i+1))

    # Add legend
    unique_shapes = list(set([G.nodes[node]['shape'] for node in G.nodes()]))
    legend_labels = [f"Block {i+1}" for i in range(len(unique_shapes))]
    for i,marker in enumerate(unique_shapes):
        axes[-1].scatter([],[], label=legend_labels[i], marker = marker, c="black")
    axes[-1].legend(loc=legend_loc)
    
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes.ravel().tolist()
    )
    cbar.set_label(r"Initial preference, $A_{0,i,m}$")

    fig.suptitle("Initial preference")
    
    plotName = fileName + "/Prints"

    f = plotName + "/SBM_network_start_preference"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_identity_matrix(fileName, Data, dpi_save,latex_bool = False):

    fig, ax = plt.subplots(figsize=(10,6))
    y_title = r"Identity, $I_{t,n}$"

    Data_preferences_trans = np.asarray(Data.history_identity_vec).T#NOW ITS person then time

    for v in range(Data.N):
        ax.plot(np.asarray(Data.history_time), Data_preferences_trans[v])
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        #ax.set_ylim(0, 1)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_identity_timeseries_matrix"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")


def plot_preference_timeseries(fileName, Data, dpi_save, latex_bool=False):

    if latex_bool:
        plt.rcParams['text.usetex'] = True
    else:
        plt.rcParams['text.usetex'] = False

    fig, axes = plt.subplots(nrows=1, ncols = Data.M,figsize=(10, 6))
    
    y_title = r"Low Carbon Preference, $A_{t,i,m}$"

    # Transpose the preference history matrix to have individuals as rows and time as columns

    Data_preferences_trans = np.asarray(Data.history_low_carbon_preference_matrix).T
    # Plot each individual's preference evolution over time
    for m in range(Data.M):
        ax = axes[m]
        for n in range(Data.N):
            ax.plot(np.asarray(Data.history_time), Data_preferences_trans[m][n])
            ax.set_xlabel(r"Time")
            ax.set_title("m = %s" % (m+1))
            #ax.set_ylabel(y_title)
            # Optionally set y-axis limits if needed
            # ax.set_ylim(0, 1)
    axes[0].set_ylabel(y_title)
    plt.tight_layout()

    # Save the plots in the specified formats
    plotName = fileName + "/Plots"
    f = plotName + "/plot_preference_timeseries_matrix"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_identity_matrix_density(fileName, Data, dpi_save, bin_num, latex_bool=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_title = r"Identity, $I_{t,n}$"


    colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
    n_bins = 100  # Discretize the interpolation into bins
    cmap_name = 'white_to_blue'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Transpose the data to get time steps on the x-axis and identity values on the y-axis
    Data_preferences_trans = np.asarray(Data.history_identity_vec).T  # Now it's time then person

    # Create a 2D histogram
    h = ax.hist2d(np.tile(np.asarray(Data.history_time), Data.N), Data_preferences_trans.flatten(), bins=[len(Data.history_time), bin_num],cmap = custom_cmap)# cmap='viridis')

    # Set the labels
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    # Add color bar to indicate the density
    plt.colorbar(h[3], ax=ax, label='Density')

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_identity_timeseries_matrix_density"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_emissions_flow_matrix(fileName, Data, dpi_save,latex_bool = False):

    fig, ax = plt.subplots(figsize=(10,6))
    y_title = r"Emissions individuals, $E_{t,i}$"

    Data_emissions_trans = np.asarray(Data.history_flow_carbon_emissions_vec).T#NOW ITS person then time

    for v in range(Data.N):
        ax.plot(np.asarray(Data.history_time), Data_emissions_trans[v])
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        #ax.set_ylim(0, 1)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_timeseries_matrix"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")




def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    dpi_save = 600,
    ) -> None: 

    Data = load_object(fileName + "/Data", "social_network")

    #print("Data EMissions", Data.total_carbon_emissions_stock)

    #node_shape_list = ["o","s","^","v"]

    #anim_save_bool = False#Need to install the saving thing
    ###PLOTS

    
    plot_identity_matrix(fileName, Data, dpi_save)
    plot_preference_timeseries(fileName, Data, dpi_save)



    bin_num= 200
    #plot_identity_matrix_density(fileName, Data, dpi_save, bin_num)
    #plot_emissions_flow_matrix(fileName, Data, dpi_save)
    #plot_emissions_individuals(fileName, Data, dpi_save)

    #plot_total_carbon_emissions_timeseries(fileName, Data, dpi_save)
    #plot_total_flow_carbon_emissions_timeseries(fileName, Data, dpi_save)
    #plot_chi(fileName, Data, dpi_save)
    #plot_omega(fileName, Data, dpi_save)
    #plot_consum_ratio(fileName, Data, dpi_save)
    #plot_L(fileName, Data, dpi_save)
    #plot_H(fileName, Data, dpi_save)
    #plot_Z_timeseries(fileName, Data, dpi_save)

    """
    if Data.network_type =="SBM":
        plot_SBM_low_carbon_preferences_timeseries(fileName, Data, dpi_save)
        plot_SBM_network_start_preferences(fileName, Data,cmap, dpi_save, node_sizes,norm_zero_one,block_markers_list,legend_loc,lines_alpha)
        plot_SBM_network_end_preferences(fileName, Data,cmap, dpi_save, node_sizes,norm_zero_one,block_markers_list,legend_loc,lines_alpha)
    else:
        plot_low_carbon_preferences_timeseries(fileName, Data, dpi_save)
        plot_network_start_preferences(fileName, Data,cmap, dpi_save, node_sizes,norm_zero_one)
        plot_network_end_preferences(fileName, Data,cmap, dpi_save, node_sizes,norm_zero_one)
    """
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
        fileName = "results/single_experiment_12_00_30__28_08_2024",
    )


