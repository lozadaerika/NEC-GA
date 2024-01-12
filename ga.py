import tsplib95
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def plot_tsp_graph(graph):
    pos = nx.get_node_attributes(graph, 'coords')
    nx.draw(graph, pos, with_labels=True, edge_color='gray', node_size=50, font_size=6)
    plt.title("TSP Graph with Best Solution")
    plt.show()

def create_tsp_graph(tsp_problem):
    graph = nx.Graph()
    for node in tsp_problem.node_coords:
        graph.add_node(node, coords=tsp_problem.node_coords[node])
    return graph

#TSP file
problem_file = "berlin52.tsp" 
tsp_problem = tsplib95.load(problem_file)

tsp_graph = create_tsp_graph(tsp_problem)

plot_tsp_graph(tsp_graph,)




