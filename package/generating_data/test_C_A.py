from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import check_other_folder


def calculate_C(A, Q, sigma):
    numerator = (A / (Q * (1 - A))) ** sigma
    denominator = ((A / (Q * (1 - A))) ** sigma) + 1
    C = numerator / denominator
    return C

def plot_A_vs_C(sigma_values, Q_values, A_range, line_style_list,colour_list):
    fig, ax = plt.subplots(constrained_layout=True)

    for i,sigma in enumerate(sigma_values):
        for j,Q in enumerate(Q_values):
            C_values = [calculate_C(A, Q, sigma) for A in A_range]
            ax.plot(A_range, C_values, label="$\sigma_m$ = %s, $Q_{t,m}$ = %s" % (sigma,Q),  linestyle= line_style_list[i], c = colour_list[j])

    ax.set_xlabel('$A_{t,i,m}$')
    ax.set_ylabel('$C_{t,i,m}$')
    #ax.set_title('A vs C for Different Sigma and Q Values')
    ax.legend()
    fig.tight_layout()

    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/C_A"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 


if __name__ == '__main__':
    
    # Example usage:
    sigma_values = [1, 2]
    sigma_values = [1, 2, 100]
    Q_values = [0.1, 0.5, 1.0]
    A_range = np.linspace(0, 1, 1000)
    line_style_list = ["solid", "dotted", "dashed", "dashdot"]
    colour_list = [ "red", "blue", "green", "yellow", "purple", "orange", "white", "black" ]

    plot_A_vs_C(sigma_values, Q_values, A_range,line_style_list,colour_list)

    plt.show()


