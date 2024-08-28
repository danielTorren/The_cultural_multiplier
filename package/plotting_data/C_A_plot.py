from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from package.resources.utility import check_other_folder


def calculate_C(A, Q, sigma):
    numerator = (A / (Q * (1 - A))) ** sigma
    denominator = ((A / (Q * (1 - A))) ** sigma) + 1
    C = numerator / denominator
    return C

def calculate_C_alt(A, Q, sigma):
    numerator = A**sigma
    denominator = A**sigma +  (Q * (1 - A))**sigma
    C = numerator / denominator
    return C

def calculate_C_expenditure(A, Q, sigma):
    numerator = A**sigma
    denominator = A**sigma +  (Q * (1 - A))**(sigma-1)
    C = numerator / denominator
    return C

def plot_A_vs_C(sigma_values, Q_values, A_range, line_style_list,colour_list):
    fig, ax = plt.subplots(constrained_layout=True)

    for i,sigma in enumerate(sigma_values):
        for j,Q in enumerate(Q_values):
            C_values = [calculate_C(A, Q, sigma) for A in A_range]
            ax.plot(A_range, C_values, label="$\sigma_m$ = %s, $\\bar{P}_{t,m}$ = %s" % (sigma,Q),  linestyle= line_style_list[i], c = colour_list[j])

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

def plot_A_vs_C_triple(sigma_values, Q_values, A_range, line_style_list,colour_list):

    fig, axes = plt.subplots(nrows = 1, ncols = len(sigma_values), constrained_layout=True, figsize= (10,5), sharey=True)

    axes[0].set_ylabel('$C_{t,i,m}$')

    for i,sigma in enumerate(sigma_values):
        axes[i].set_title(("$\sigma_m$ = %s") % (sigma))
        for j,Q in enumerate(Q_values):
            C_values = [calculate_C(A, Q, sigma) for A in A_range]
            #print(a)
            #quit()
            axes[i].plot(A_range, C_values, label = "$Q_m$ = %s" % (Q), linestyle= line_style_list[i])## c = colour_list[j]
        axes[i].set_xlabel('$A_{t,i,m}$')
        axes[i].legend(loc = "lower right")#"lower right"

    #ax.set_title('A vs C for Different Sigma and Q Values')
    
    #fig.tight_layout()

    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/C_A_triple"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_A_vs_C_triple_alt(sigma_values, Q_values, A_range, line_style_list,colour_list):

    fig, axes = plt.subplots(nrows = 1, ncols = len(sigma_values), constrained_layout=True, figsize= (10,5), sharey=True)

    axes[0].set_ylabel('$C_{t,i,m}$')

    for i,sigma in enumerate(sigma_values):
        axes[i].set_title(("$\sigma_m$ = %s") % (sigma))
        for j,Q in enumerate(Q_values):
            C_values = [calculate_C_alt(A, Q, sigma) for A in A_range]
            #print(a)
            #quit()
            axes[i].plot(A_range, C_values, label = "$\\bar{P_m}$ = %s" % (Q), linestyle= line_style_list[i])## c = colour_list[j]
        axes[i].set_xlabel('$A_{t,i,m}$')
        axes[i].legend(loc = "lower right")#"lower right"

    #ax.set_title('A vs C for Different Sigma and Q Values')

    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/C_A_triple_alt"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")  

def plot_A_vs_C_triple_expenditure(sigma_values, Q_values, A_range, line_style_list,colour_list):

    fig, axes = plt.subplots(nrows = 1, ncols = len(sigma_values), constrained_layout=True, figsize= (10,5), sharey=True)

    axes[0].set_ylabel('$C_{t,i,m}$')

    for i,sigma in enumerate(sigma_values):
        axes[i].set_title(("$\sigma_m$ = %s") % (sigma))
        for j,Q in enumerate(Q_values):
            C_values = [calculate_C_expenditure(A, Q, sigma) for A in A_range]
            #print(a)
            #quit()
            axes[i].plot(A_range, C_values, label = "$Q_m$ = %s" % (Q), linestyle= line_style_list[i])## c = colour_list[j]
        axes[i].set_xlabel('$A_{t,i,m}$')
        axes[i].legend(loc = "lower right")#"lower right"

    fig.suptitle("Expenditure")
    
    #fig.tight_layout()

    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/C_A_triple_alt"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")  

if __name__ == '__main__':
    
    # Example usage:
    #sigma_values = [1, 2]
    sigma_values = [1, 2,10]
    Q_values = [0.5, 0.75, 1.0]
    A_range = np.linspace(0, 1, 1000)
    line_style_list = ["solid", "dotted", "dashed", "dashdot","solid", "dotted"]
    colour_list = [ "red", "blue", "green", "yellow", "purple", "orange", "white", "black" ]

    #plot_A_vs_C(sigma_values, Q_values, A_range,line_style_list,colour_list)
    plot_A_vs_C_triple_alt(sigma_values, Q_values, A_range,line_style_list,colour_list)
    #plot_A_vs_C_triple_expenditure(sigma_values, Q_values, A_range,line_style_list,colour_list)
    plt.show()


