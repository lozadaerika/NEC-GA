import tsplib95
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def plot_tsp_graph(graph, solution):
    pos = nx.get_node_attributes(graph, 'coords')
    edges = list(graph.edges())

    nx.draw(graph, pos, with_labels=True, edge_color='gray', node_size=50, font_size=6)

    #Highlight solution
    solution_edges = [(solution[i], solution[i + 1]) for i in range(len(solution) - 1)]
    solution_edges.append((solution[-1], solution[0]))
    nx.draw_networkx_edges(graph, pos, edgelist=solution_edges, edge_color='red', width=2)

    plt.title("TSP Graph with Best Solution")
    plt.show()

def calculate_euclidean_distance(coord1, coord2):
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5

def create_tsp_graph(tsp_problem):
    graph = nx.Graph()

    for node in tsp_problem.node_coords:
        graph.add_node(node, coords=tsp_problem.node_coords[node])

    for edge in tsp_problem.get_edges():
        if edge[0] != edge[1]: 
            # Exclude edges to the same node
            coord1 = tsp_problem.node_coords[edge[0]]
            coord2 = tsp_problem.node_coords[edge[1]]
            weight = calculate_euclidean_distance(coord1, coord2)
            graph.add_edge(edge[0], edge[1], weight=weight)

    return graph

def calculate_total_distance(solution, graph):
    total_distance = 0   
    for i in range(len(solution) - 1):
        total_distance += graph[solution[i]][solution[i + 1]]['weight']
    # Return to the start
    total_distance += graph[solution[-1]][solution[0]]['weight'] 
    return total_distance

#TSP file
problem_file = "berlin52.tsp" 
tsp_problem = tsplib95.load(problem_file)

tsp_graph = create_tsp_graph(tsp_problem)

plot_tsp_graph(tsp_graph,)




