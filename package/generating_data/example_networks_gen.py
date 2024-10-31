# imports
from package.resources.utility import (
    createFolder, 
    produce_name_datetime
)
import matplotlib.pyplot as plt
import networkx as nx

def prod_pos(layout_type: str, network: nx.Graph) -> nx.Graph:

    if layout_type == "circular":
        pos_identity_network = nx.circular_layout(network)
    elif layout_type == "spring":
        pos_identity_network = nx.spring_layout(network)
    elif layout_type == "kamada_kawai":
        pos_identity_network = nx.kamada_kawai_layout(network)
    elif layout_type == "planar":
        pos_identity_network = nx.planar_layout(network)
    else:
        raise Exception("Invalid layout given")

    return pos_identity_network

def plot_network_examples(    
        fileName
    ):

    fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(9,3.2), constrained_layout = True)


    G_1 = nx.watts_strogatz_graph(n=50, k=5, p=0.05, seed=1)  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
    G_2 = nx.stochastic_block_model(sizes=[50,50], p=[[0.1,0.005],[0.005,0.1]], seed=1)
    G_3 = nx.barabasi_albert_graph(n=50, m=3, seed= 1)


    pos_1 = prod_pos("circular", G_1)
    pos_2 = nx.spring_layout(G_2,seed=1)  # You can use other layout algorithms
    pos_3 = prod_pos("circular", G_3)#nx.spring_layout(G_3,seed=1)  # You can use other layout algorithms

    node_size_val = 70
     
    nx.draw(
        G_1,
        ax=axes[0],
        pos=pos_1,
        node_size=node_size_val,
        edgecolors="black",
    )

    nx.draw(
        G_2,
        ax=axes[1],
        pos=pos_2,
        node_size=node_size_val,
        edgecolors="black",
    )

    nx.draw(
        G_3,
        ax=axes[2],
        pos=pos_3,
        node_size=node_size_val,
        edgecolors="black"
    )

    axes[0].set_title("Small-World")
    axes[1].set_title("Stochastic Block Model")
    axes[2].set_title("Scale-Free")

    plotName = fileName + "/Prints"

    f = plotName + "/network_example"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    ) -> None: 

    root = "example_graphs"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    createFolder(fileName)

    plot_network_examples(fileName)
  
    plt.show()

if __name__ == '__main__':
    plots = main(
        fileName = "results/single_experiment_11_02_44__19_02_2024",
    )


