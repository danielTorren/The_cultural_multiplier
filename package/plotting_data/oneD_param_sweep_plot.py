"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from package.resources.utility import load_object
from package.resources.plot import (
    plot_end_points_emissions,
    plot_end_points_emissions_scatter,
    plot_end_points_emissions_lines,
    plot_end_points_emissions_scatter_gini,
    plot_end_points_emissions_lines_gini
)
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import seaborn as sns

def plot_stacked_preferences(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(data_list),ncols=data_list[0].M, sharex="col", constrained_layout = True,figsize=(14, 7))

    for i, data in enumerate(data_list):
        #axes[i][0].set_title(property_varied_title + " = " + str(round(property_values_list[i],3)))
        for v in range(data.N):
            data_indivdiual = np.asarray(data.agent_list[v].history_low_carbon_preferences)
            #print("data_indivdiual",data_indivdiual,len(data_indivdiual))
            #quit()
            for j in range(data.M):
                #if i == 0:
                #axes[0][j].set_title("$\sigma_{%s} = %s$" % (j,data.low_carbon_substitutability_array[j]))
                axes[i][j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )

    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(data_list[0].low_carbon_substitutability_array[i],3))) for i in range(len(data_list[0].low_carbon_substitutability_array))]
    rows = ["%s=%s" % (property_varied_title,str(round(val,3))) for val in property_values_list]

    #print(cols)
    #print(rows)
    pad = 2 # in points
    #"""
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')
    #"""
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='small', ha='right', va='center',rotation=90)
        
    fig.supxlabel(r"Time")
    fig.supylabel(r"Low carbon preference")
    
    plotName = fileName + "/Prints"

    f = plotName + "/timeseries_preference_stacked_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_stacked_chi_m(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(data_list),ncols=data_list[0].M, sharex="col", constrained_layout = True,figsize=(14, 7))

    for i, data in enumerate(data_list):
        #axes[i][0].set_title(property_varied_title + " = " + str(round(property_values_list[i],3)))
        for v in range(data.N):
            data_indivdiual = np.asarray(data.agent_list[v].history_chi_m)
            for j in range(data.M):
                #if i == 0:
                #axes[0][j].set_title("$\sigma_{%s} = %s$" % (j,data.low_carbon_substitutability_array[j]))
                axes[i][j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )

    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(data_list[0].low_carbon_substitutability_array[i],3))) for i in range(len(data_list[0].low_carbon_substitutability_array))]
    rows = ["%s=%s" % (property_varied_title,str(round(val,3))) for val in property_values_list]

    #print(cols)
    #print(rows)
    pad = 2 # in points
    #"""
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')
    #"""

    
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='small', ha='right', va='center',rotation=90)
        
    fig.supxlabel(r"Time")
    fig.supylabel("$\chi$")
    
    plotName = fileName + "/Prints"

    f = plotName + "/chi_stacked_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_stacked_omega_m(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(data_list),ncols=data_list[0].M, sharex="col", constrained_layout = True,figsize=(14, 7))

    for i, data in enumerate(data_list):
        #axes[i][0].set_title(property_varied_title + " = " + str(round(property_values_list[i],3)))
        for v in range(data.N):
            data_indivdiual = np.asarray(data.agent_list[v].history_omega_m)
            for j in range(data.M):
                #if i == 0:
                #axes[0][j].set_title("$\sigma_{%s} = %s$" % (j,data.low_carbon_substitutability_array[j]))
                axes[i][j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )
                #axes[i][j].set_ylim(0,2)

    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(data_list[0].low_carbon_substitutability_array[i],3))) for i in range(len(data_list[0].low_carbon_substitutability_array))]
    rows = ["%s=%s" % (property_varied_title,str(round(val,3))) for val in property_values_list]

    #print(cols)
    #print(rows)
    pad = 2 # in points
    #"""
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')
    #"""
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='small', ha='right', va='center',rotation=90)
        
    fig.supxlabel(r"Time")
    fig.supylabel("$\Omega$")
    
    plotName = fileName + "/Prints"

    f = plotName + "/omega_stacked_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_stacked_H_m(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(data_list),ncols=data_list[0].M, sharex="col", constrained_layout = True,figsize=(14, 7))

    for i, data in enumerate(data_list):
        #axes[i][0].set_title(property_varied_title + " = " + str(round(property_values_list[i],3)))
        for v in range(data.N):
            data_indivdiual = np.asarray(data.agent_list[v].history_H_m)
            for j in range(data.M):
                #if i == 0:
                #axes[0][j].set_title("$\sigma_{%s} = %s$" % (j,data.low_carbon_substitutability_array[j]))
                axes[i][j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )

    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(data_list[0].low_carbon_substitutability_array[i],3))) for i in range(len(data_list[0].low_carbon_substitutability_array))]
    rows = ["%s=%s" % (property_varied_title,str(round(val,3))) for val in property_values_list]

    #print(cols)
    #print(rows)
    pad = 2 # in points
    #"""
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')
    #"""

    

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='small', ha='right', va='center',rotation=90)
        
    fig.supxlabel(r"Time")
    fig.supylabel("$H_m$")
    
    plotName = fileName + "/Prints"

    f = plotName + "/H_stacked_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_stacked_L_m(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(data_list),ncols=data_list[0].M, sharex="col", constrained_layout = True,figsize=(14, 7))

    for i, data in enumerate(data_list):
        #axes[i][0].set_title(property_varied_title + " = " + str(round(property_values_list[i],3)))
        for v in range(data.N):
            data_indivdiual = np.asarray(data.agent_list[v].history_L_m)
            for j in range(data.M):
                #if i == 0:
                #axes[0][j].set_title("$\sigma_{%s} = %s$" % (j,data.low_carbon_substitutability_array[j]))
                axes[i][j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )

    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(data_list[0].low_carbon_substitutability_array[i],3))) for i in range(len(data_list[0].low_carbon_substitutability_array))]
    rows = ["%s=%s" % (property_varied_title,str(round(val,3))) for val in property_values_list]

    #print(cols)
    #print(rows)
    pad = 2 # in points
    #"""
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')
    #"""

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='small', ha='right', va='center',rotation=90)
        
    fig.supxlabel(r"Time")
    fig.supylabel("$L_m$")
    
    plotName = fileName + "/Prints"

    f = plotName + "/L_stacked_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_stacked_preferences_averages(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(data_list),ncols=data_list[0].M, sharex="col", constrained_layout = True,figsize=(14, 7))


    # I need to get the data into the shape [property run, M,time,N], then i can take the average and media of the last value

    for i, data in enumerate(data_list):
        data_store = []#shape [N,time,M]
        for v in range(data.N):
            data_store.append(np.asarray(data.agent_list[v].history_low_carbon_preferences)) # thing being appended this has shape [time, M]
        data_array = np.asarray(data_store)
        data_trans = data_array.transpose(2,1,0)#will now be [M,time,N]

        for j in range(data.M):
            data_mean = np.mean(data_trans[j], axis=1)
            data_median = np.median(data_trans[j], axis=1)
            axes[i][j].plot(np.asarray(data.history_time),data_mean, label = "mean")
            axes[i][j].plot(np.asarray(data.history_time),data_median, label = "median")
            axes[i][j].legend()

    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(data_list[0].low_carbon_substitutability_array[i],3))) for i in range(len(data_list[0].low_carbon_substitutability_array))]
    rows = ["%s=%s" % (property_varied_title,str(round(val,3))) for val in property_values_list]

    #print(cols)
    #print(rows)
    pad = 2 # in points
    #"""
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')
    #"""
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='small', ha='right', va='center',rotation=90)
        
    fig.supxlabel(r"Time")
    fig.supylabel(r"Low carbon preference")
    
    plotName = fileName + "/Prints"

    f = plotName + "/averages_timeseries_preference_stacked_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_stacked_omega_m(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(data_list),ncols=data_list[0].M, sharex="col", constrained_layout = True,figsize=(14, 7))


    # I need to get the data into the shape [property run, M,time,N], then i can take the average and media of the last value

    for i, data in enumerate(data_list):
        data_store = []#shape [N,time,M]
        for v in range(data.N):
            data_store.append(np.asarray(data.agent_list[v].history_omega_m)) # thing being appended this has shape [time, M]
        data_array = np.asarray(data_store)
        data_trans = data_array.transpose(2,1,0)#will now be [M,time,N]

        for j in range(data.M):
            data_mean = np.mean(data_trans[j], axis=1)
            data_median = np.median(data_trans[j], axis=1)
            axes[i][j].plot(np.asarray(data.history_time),data_mean, label = "mean")
            axes[i][j].plot(np.asarray(data.history_time),data_median, label = "median")
            axes[i][j].legend()

    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(data_list[0].low_carbon_substitutability_array[i],3))) for i in range(len(data_list[0].low_carbon_substitutability_array))]
    rows = ["%s=%s" % (property_varied_title,str(round(val,3))) for val in property_values_list]

    #print(cols)
    #print(rows)
    pad = 2 # in points
    #"""
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')
    #"""
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='small', ha='right', va='center',rotation=90)
        
    fig.supxlabel(r"Time")
    fig.supylabel("$\Omega$")
    
    plotName = fileName + "/Prints"

    f = plotName + "/averages_timeseries_omega_stacked_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def multi_data_and_col_fixed_animation_distribution(fileName, Data_run_list, property_plot, x_axis_label, direction,property_varied_title,property_values_list, dpi_save, save_bool):


    cols = ["%s=%s" % (property_varied_title,str(round(val,3))) for val in property_values_list]

    data_store = []

    min_lim = 0#change this if they arent [0,1] variable such as utility
    max_lim = 1

    for data_sim in Data_run_list:
        time_series = data_sim.history_time
        data_list = []  
        for v in range(data_sim.N):
            data_list.append(np.asarray(eval("data_sim.agent_list[v].%s" % property_plot)))
        
        data_matrix = np.asarray(data_list)#[N, T, M]

        reshaped_array = np.transpose(data_matrix, (2, 1, 0))#[M, T, N] SO COOL THAT YOU CAN SPECIFY REORDING WITH TRANSPOSE!

        data_pd_list = []
        for data_matrix_T in reshaped_array:
            # Example data
            data = pd.DataFrame({
                'Time': time_series,
                'Distribution': data_matrix_T.tolist()
            })
            data_pd_list.append(data)

        data_store.append(data_pd_list)

    # Create a figure and axis for the animation
    fig, axes = plt.subplots(figsize=(10, 5), nrows=1,ncols=len(Data_run_list), sharey=True, constrained_layout = True)
    sns.set_style("whitegrid")

    pad = 5

    for k, ax in enumerate(axes.flat):
        ax.annotate(cols[k], xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')#add rthe label
            
        data_pd_list = data_store[k]
        # Set up the initial KDE plot
        if direction == "y":
            for i,data in enumerate(data_pd_list):
                initial_kde = sns.kdeplot(ax = ax,y = data['Distribution'].iloc[0], label="$\sigma_{%s}$ = %s" % (i+1,data_sim.low_carbon_substitutability_array[i] ))
            ax.set_ylabel(x_axis_label)
            ax.set_xlabel('Density')
            ax.set_ylim(min_lim,max_lim)
        else:
            for i,data in enumerate(data_pd_list):
                initial_kde = sns.kdeplot(ax = ax,x = data['Distribution'].iloc[0], label="$\sigma_{%s}$ = %s" % (i+1,data_sim.low_carbon_substitutability_array[i] ))#color='b'
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel('Density')
            ax.set_xlim(min_lim,max_lim)
        #ax.set_title(f"Distribution Over Time - Time {data['Time'].iloc[0]}")
        ax.legend(loc='upper right')

    
    fig.suptitle(f"Steps: {data_store[0][0]['Time'].iloc[0]}/{data_store[0][0]['Time'].iloc[-1]}")

    def update(frame):
        fig.suptitle(f"Steps: {data_store[0][0]['Time'].iloc[frame]}/{data_store[0][0]['Time'].iloc[-1]}")
        
        for k, ax in enumerate(axes.flat):
            ax.cla()  # Clear the current axes
            
            ax.annotate(cols[k], xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='small', ha='center', va='baseline')  # Add the top label

            data_pd_list = data_store[k]  # Get the right data

            if direction == "y":
                for i, data in enumerate(data_pd_list):
                    kde = sns.kdeplot(ax=ax, y=data['Distribution'].iloc[frame],
                                    label="$\sigma_{%s}$ = %s" % (i + 1, data_sim.low_carbon_substitutability_array[i]))
                ax.set_ylabel(x_axis_label)
                ax.set_ylim(min_lim, max_lim)
                ax.set_xlabel('Density')
            else:
                for i, data in enumerate(data_pd_list):
                    kde = sns.kdeplot(ax=ax, x=data['Distribution'].iloc[frame],
                                    label="$\sigma_{%s}$ = %s" % (i + 1, data_sim.low_carbon_substitutability_array[i]))
                ax.set_xlabel(x_axis_label)
                ax.set_xlim(min_lim, max_lim)
                ax.set_ylabel('Density')
            ax.legend(loc='upper right')


    # Define the update function to animate the KDE plot
    def update_og(frame):
        
        fig.suptitle(f"Steps: {data_store[0][0]['Time'].iloc[frame]}/{data_store[0][0]['Time'].iloc[-1]}")
        for k, ax in enumerate(axes.flat):
            ax.clear()

            ax.annotate(cols[k], xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')#add the top label
                
            data_pd_list = data_store[k]# get the right data
            
            if direction == "y":
                for i,data in enumerate(data_pd_list):
                    kde = sns.kdeplot(ax = ax, y=data['Distribution'].iloc[frame], label="$\sigma_{%s}$ = %s" % (i+1,data_sim.low_carbon_substitutability_array[i] ))
                ax.set_ylabel(x_axis_label)
                ax.set_ylim(min_lim,max_lim)
                ax.set_xlabel('Density')
            else:
                for i,data in enumerate(data_pd_list):
                    kde = sns.kdeplot(ax = ax, x=data['Distribution'].iloc[frame], label="$\sigma_{%s}$ = %s" % (i+1,data_sim.low_carbon_substitutability_array[i] ))
                ax.set_xlabel(x_axis_label)
                ax.set_xlim(min_lim,max_lim)
                ax.set_ylabel('Density')
            ax.legend(loc='upper right')

    animation = FuncAnimation(fig, update, frames=len(data_store[0][0]["Time"]), repeat_delay=100,interval=0.1)

    plt.show()

    return animation

"""
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
"""


def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    dpi_save = 600,
    latex_bool = 0,
    PLOT_TYPE = 1
    ) -> None: 

    ############################
    
    #print(base_params)
    #quit()

    base_params = load_object(fileName + "/Data", "base_params")
    var_params  = load_object(fileName + "/Data" , "var_params")
    property_values_list = load_object(fileName + "/Data", "property_values_list")

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption",
    property_min = var_params["property_min"]#0,
    property_max = var_params["property_max"]#1,
    property_reps = var_params["property_reps"]#10,
    property_varied_title = var_params["property_varied_title"]

    if PLOT_TYPE == 5:
        data_list = load_object(fileName + "/Data", "data_list")
    else:
        emissions_array = load_object(fileName + "/Data", "emissions_array")
        
    if PLOT_TYPE == 1:
        reduc_emissions_array = emissions_array[:-1]
        reduc_property_values_list = property_values_list[:-1]
        #plot how the emission change for each one
        plot_end_points_emissions(fileName, reduc_emissions_array, property_varied_title, property_varied, reduc_property_values_list, dpi_save)
    elif PLOT_TYPE == 2:
        plot_end_points_emissions(fileName, emissions_array, "Preference to consumption ratio, $\\mu$", property_varied, property_values_list, dpi_save)
    elif PLOT_TYPE == 3:
        gini_array = load_object(fileName + "/Data", "gini_array")

        plot_end_points_emissions(fileName, emissions_array, "Budget inequality (Pareto distribution constant)", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_scatter_gini(fileName, emissions_array, "Initial Gini index", property_varied, property_values_list,gini_array, dpi_save)
        plot_end_points_emissions_lines_gini(fileName, emissions_array, "Initial Gini index", property_varied, property_values_list,gini_array, dpi_save)
    elif PLOT_TYPE == 4:
        #gini_array = load_object(fileName + "/Data", "gini_array")
        plot_end_points_emissions(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_scatter(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_lines(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
    if PLOT_TYPE == 5:
        # look at splitting of the last behaviour with preference dissonance
        #property_varied_title = "$\sigma_A$"
        plot_stacked_preferences(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        plot_stacked_chi_m(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        plot_stacked_omega_m(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        plot_stacked_H_m(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        plot_stacked_L_m(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)

        plot_stacked_preferences_averages(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        plot_stacked_omega_m(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        #anim_save_bool = False
        #multi_data_and_col_fixed_animation_distribution(fileName, data_list, "history_low_carbon_preferences","Low carbon Preferences","y", property_varied_title,property_values_list,dpi_save,anim_save_bool)
        #DONT PUT ANYTHING MORE PLOTS AFTER HERE DUE TO ANIMATION 
    else:
        plot_end_points_emissions(fileName, emissions_array, property_varied_title, property_varied, property_values_list, dpi_save)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/one_param_sweep_multi_18_52_13__14_11_2023",
        PLOT_TYPE = 5
    )

