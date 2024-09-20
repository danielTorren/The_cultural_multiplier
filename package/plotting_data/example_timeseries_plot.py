"""Plots a single simulation to produce data which is saved and plotted 

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import ( 
    load_object,
)

def plot_identity_matrix_2(fileName, Data_1, Data_2):


    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10,6), sharey=True)

    Data_preferences_trans_1 = np.asarray(Data_1.history_identity_vec).T#NOW ITS person then time
    Data_preferences_trans_2 = np.asarray(Data_2.history_identity_vec).T#NOW ITS person then time

    for v in range(Data_1.N):
        axes[0].plot(np.asarray(Data_1.history_time), Data_preferences_trans_1[v])
        axes[1].plot(np.asarray(Data_2.history_time), Data_preferences_trans_2[v])
        
    axes[0].set_xlabel(r"Timestep")
    axes[1].set_xlabel(r"Timestep")

    axes[0].set_title(r"Carbon tax, $\tau = 0$")
    axes[1].set_title(r"Carbon tax, $\tau = 0.15$")

    axes[0].set_ylabel(r"Environmental identity, $I_{t,i}$")

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_example_timerseries"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    ) -> None: 

    Data_no = load_object(fileName + "/Data", "Data_no")
    Data_high = load_object(fileName + "/Data", "data_high")
    
    plot_identity_matrix_2(fileName, Data_no,  Data_high)

    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/single_experiment_11_02_44__19_02_2024",
    )


