"""Plot multiple single simulations varying a single parameter

Created: 10/10/2022
"""

# imports
from cProfile import label
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
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable

def plot_stacked_preferences(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save):


    if data_list[0].M == 1:
        fig, axes = plt.subplots(nrows=1,ncols=len(data_list), sharey=True, constrained_layout = True,figsize=(14, 7))

        for i, data in enumerate(data_list):
            axes[i].set_title(property_varied_title + " = " + str(round(property_values_list[i],3)))
            for v in range(data.N):
                data_indivdiual = np.asarray(data.agent_list[v].history_low_carbon_preferences)
                axes[i].plot(
                        np.asarray(data.history_time),
                        data_indivdiual
                    )
                
        fig.supxlabel(r"Time")
        axes[0].set_ylabel(r"Low carbon preference, $A_{t,i,m}$")
    else:
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
        rows = ["%s=%s" % (property_varied_title,str(round(val,4))) for val in property_values_list]

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
    rows = ["%s=%s" % (property_varied_title,str(round(val,4))) for val in property_values_list]

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
    rows = ["%s=%s" % (property_varied_title,str(round(val,4))) for val in property_values_list]

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
    rows = ["%s=%s" % (property_varied_title,str(round(val,4))) for val in property_values_list]

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
    rows = ["%s=%s" % (property_varied_title,str(round(val,4))) for val in property_values_list]

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
    rows = ["%s=%s" % (property_varied_title,str(round(val,4))) for val in property_values_list]

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
    rows = ["%s=%s" % (property_varied_title,str(round(val,4))) for val in property_values_list]

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

def plot_utility(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(data_list),ncols=data_list[0].M, constrained_layout = True,figsize=(14, 7))

    # I need to get the data into the shape [property run, M,time,N], then i can take the average and media of the last value

    for i, data in enumerate(data_list):
        data_store = []#shape [N,time,M]
        for v in range(data.N):
            data_store.append(np.asarray(data.agent_list[v].history_pseudo_utility)) # thing being appended this has shape [time, M]
        data_array = np.asarray(data_store)
        data_trans = data_array.transpose(2,1,0)#will now be [M,time,N]

        for j in range(data.M):
            data_mean = np.mean(data_trans[j], axis=1)
            data_median = np.median(data_trans[j], axis=1)
            axes[i][j].plot(np.asarray(data.history_time),data_mean,linestyle = "dashed", label = "mean")
            axes[i][j].plot(np.asarray(data.history_time),data_median,linestyle = "dotted", label = "median")
            axes[i][j].legend()
            
            axes[i][j].plot(
                    np.asarray(data.history_time),
                    data_trans[j]
                )

    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(data_list[0].low_carbon_substitutability_array[i],3))) for i in range(len(data_list[0].low_carbon_substitutability_array))]
    rows = ["%s=%s" % (property_varied_title,str(round(val,4))) for val in property_values_list]

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
    fig.supylabel("pseudo utility")
    
    plotName = fileName + "/Prints"

    f = plotName + "/history_pseudo_utility_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_stacked_total_quant(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save):

    fig, axes = plt.subplots(nrows=len(data_list),ncols=data_list[0].M, sharex="col", constrained_layout = True,figsize=(14, 7))

    for i, data in enumerate(data_list):
        #axes[i][0].set_title(property_varied_title + " = " + str(round(property_values_list[i],3)))
        for v in range(data.N):
            a = np.asarray(data.agent_list[v].history_L_m)
            b = np.asarray(data.agent_list[v].history_H_m)
            data_indivdiual = np.asarray(data.agent_list[v].history_L_m) + np.asarray(data.agent_list[v].history_H_m)
            for j in range(data.M):
                #if i == 0:
                #axes[0][j].set_title("$\sigma_{%s} = %s$" % (j,data.low_carbon_substitutability_array[j]))
                axes[i][j].plot(
                    np.asarray(data.history_time),
                    data_indivdiual[:,j]
                )

    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(data_list[0].low_carbon_substitutability_array[i],3))) for i in range(len(data_list[0].low_carbon_substitutability_array))]
    rows = ["%s=%s" % (property_varied_title,str(round(val,4))) for val in property_values_list]

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
    fig.supylabel("$Q_m = H_m + L_m$")
    
    plotName = fileName + "/Prints"

    f = plotName + "/plot_stacked_total_quant_%s" %(property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings
    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection
    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)
    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def plot_stuff_one_individual(fileName, data_list,property_values_list, property_varied, property_varied_title, dpi_save,indiv_index):

    rows = 4

    fig, axes = plt.subplots(nrows=rows,ncols=data_list[0].M, sharex="col", constrained_layout = True,figsize=(14, 7))

    c = property_values_list
    cmap = plt.get_cmap("cividis")

    for i, data in enumerate(data_list):#loop through different property values
        A = np.asarray(data.agent_list[indiv_index].history_low_carbon_preferences).T
        H = np.asarray(data.agent_list[indiv_index].history_H_m).T
        L = np.asarray(data.agent_list[indiv_index].history_L_m).T
        Q = (np.asarray(data.agent_list[indiv_index].history_L_m) + np.asarray(data.agent_list[indiv_index].history_H_m)).T
        
        for j in range(data.M):
            
            axes[0][j].plot(np.asarray(data.history_time),A[j],  color=cmap(c[i]),label = "$\tau = %s$" % (str(round(property_values_list[i],4))))
            axes[1][j].plot(np.asarray(data.history_time),H[j],  color=cmap(c[i]),label = "$\tau = %s$" % (str(round(property_values_list[i],4))))
            axes[2][j].plot(np.asarray(data.history_time),L[j],  color=cmap(c[i]),label = "$\tau = %s$" % (str(round(property_values_list[i],4))))
            axes[3][j].plot(np.asarray(data.history_time),Q[j],  color=cmap(c[i]),label = "$\tau = %s$" % (str(round(property_values_list[i],4))))
    
    axes[0][0].set_ylabel("A")
    axes[1][0].set_ylabel("H")
    axes[2][0].set_ylabel("L")
    axes[3][0].set_ylabel("Q")

    # Create a ScalarMappable to map values to colors
    norm = plt.Normalize(min(property_values_list), max(property_values_list))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You need to set an array for the mappable

    # Add a colorbar
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", pad=0.02)
    cbar.set_label(property_varied_title)

    cols = ["$\sigma_{%s}=%s$" % (i+1,str(round(data_list[0].low_carbon_substitutability_array[i],3))) for i in range(len(data_list[0].low_carbon_substitutability_array))]
    #rows = ["%s=%s" % (property_varied_title,str(round(val,4))) for val in property_values_list]

    #print(cols)
    #print(rows)
    pad = 2 # in points
    #"""
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')
    #"""
    #for ax, row in zip(axes[:,0], rows):
    #    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
    #                xycoords=ax.yaxis.label, textcoords='offset points',
    #                size='small', ha='right', va='center',rotation=90)
        
    fig.supxlabel(r"Time")
    fig.suptitle("Agent %s" % (indiv_index))
    #fig.supylabel("$Q_m = H_m + L_m$")
    
    plotName = fileName + "/Prints"

    f = plotName + "/plot_stuff_one_individual_%s_%s" %(property_varied,indiv_index)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

        

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
    #print(base_params)
    #quit()
    var_params  = load_object(fileName + "/Data" , "var_params")
    property_values_list = load_object(fileName + "/Data", "property_values_list")

    property_varied = var_params["property_varied"]#"ratio_preference_or_consumption_state",
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

        plot_end_points_emissions(fileName, emissions_array, "expenditure inequality (Pareto distribution constant)", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_scatter_gini(fileName, emissions_array, "Initial Gini index", property_varied, property_values_list,gini_array, dpi_save)
        plot_end_points_emissions_lines_gini(fileName, emissions_array, "Initial Gini index", property_varied, property_values_list,gini_array, dpi_save)
    elif PLOT_TYPE == 4:
        #gini_array = load_object(fileName + "/Data", "gini_array")
        plot_end_points_emissions(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_scatter(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_lines(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
    if PLOT_TYPE == 5:

        """
        emissions_array= np.asarray([[x.total_carbon_emissions_stock] for x in data_list])
        plot_end_points_emissions(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_scatter(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
        plot_end_points_emissions_lines(fileName, emissions_array, "redistribution val", property_varied, property_values_list, dpi_save)
        """

        # look at splitting of the last behaviour with preference dissonance
        #property_varied_title = "$\sigma_A$"
        plot_stacked_preferences(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        #plot_stacked_chi_m(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        #plot_stacked_omega_m(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        #plot_stacked_H_m(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        #plot_stacked_L_m(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        #plot_stacked_total_quant(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        #plot_stacked_preferences_averages(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        #plot_stacked_omega_m(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)
        #plot_utility(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save)

        #plot_stuff_one_individual(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save, 1)
        #plot_stuff_one_individual(fileName,data_list,property_values_list, property_varied, property_varied_title, dpi_save, 15)
        
        #anim_save_bool = False
        #multi_data_and_col_fixed_animation_distribution(fileName, data_list, "history_low_carbon_preferences","Low carbon Preferences","y", property_varied_title,property_values_list,dpi_save,anim_save_bool)
        #DONT PUT ANYTHING MORE PLOTS AFTER HERE DUE TO ANIMATION 
    else:
        plot_end_points_emissions(fileName, emissions_array, property_varied_title, property_varied, property_values_list, dpi_save)
    
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName= "results/one_param_sweep_multi_15_42_27__01_12_2023",
        PLOT_TYPE = 5
    )

