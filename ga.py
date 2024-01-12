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
        if solution[i] in graph  and solution[i + 1] in graph[solution[i]]:
            total_distance += graph[solution[i]][solution[i + 1]]['weight']
    # Return to the start
    total_distance += graph[solution[-1]][solution[0]]['weight'] 
    return total_distance

def always_one_mutation(solution):
    #Select random gene
    mutation_point = random.randint(0, len(solution) - 1)
    #Select random value
    ramdon_value = random.randint(0, len(solution) - 1)
    #Replace gene with random value
    solution[mutation_point] = ramdon_value
    return solution

def independent_gene_mutation(solution, mutation_probability=0.1):
    for i in range(len(solution)):
        if random.uniform(0, 1) < mutation_probability:
            solution[i] = 1 - solution[i]  # Flip 0 to 1 or 1 to 0
    return solution

def one_point_crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = parent1[crossover_point:] + [city for city in parent2 if city not in parent1[crossover_point:]]
    return child

def two_point_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = parent1[start:end] + [city for city in parent2 if city not in parent1[start:end]] + parent1[end:]
    return child

def uniform_crossover(parent1, parent2):
    child = [city1 if random.choice([True, False]) else city2 for city1, city2 in zip(parent1, parent2)]
    return child

#TSP file
problem_file = "berlin52.tsp" 
tsp_problem = tsplib95.load(problem_file)

tsp_graph = create_tsp_graph(tsp_problem)

plot_tsp_graph(tsp_graph,)




