from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import check_other_folder

def calculate_C_alt(A, Q, sigma):
    numerator = A**sigma
    denominator = A**sigma +  (Q * (1 - A))**sigma
    C = numerator / denominator
    return C

def plot_A_vs_C_triple_alt(sigma_values, Q_values, A_range, line_style_list,colour_list):

    fig, axes = plt.subplots(nrows = 1, ncols = len(sigma_values), constrained_layout=True, figsize= (10,5), sharey=True)

    axes[0].set_ylabel('Proportion Low-carbon consumption, $C_{t,i,m}$', fontsize="12")

    for i,sigma in enumerate(sigma_values):
        axes[i].set_title(("Substitutability, $\sigma_m$ = %s") % (sigma), fontsize="12")
        for j,Q in enumerate(Q_values):
            C_values = [calculate_C_alt(A, Q, sigma) for A in A_range]
            axes[i].plot(A_range, C_values, label = "$\\bar{P_m}$ = %s" % (Q), linestyle= line_style_list[i])## c = colour_list[j]
        axes[i].grid()
        
    fig.supxlabel('Low-carbon preference, $A_{t,i,m}$', fontsize="12")
    #ax.set_title('A vs C for Different Sigma and Q Values')
    axes[2].legend(loc = "lower right")#"lower right"
    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/C_A_triple_alt"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")  
    plt.show()


if __name__ == '__main__':
    sigma_values = [1, 2,10]
    Q_values = [0.5, 0.75, 1.0]
    A_range = np.linspace(0, 1, 1000)
    line_style_list = ["solid", "dotted", "dashed", "dashdot","solid", "dotted"]
    colour_list = [ "red", "blue", "green", "yellow", "purple", "orange", "white", "black" ]

    plot_A_vs_C_triple_alt(sigma_values, Q_values, A_range,line_style_list,colour_list)



