import tsplib95
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def plot_tsp_graph(graph, solution):
    #Plot graph
    pos = nx.get_node_attributes(graph, 'coords')
    nx.draw(graph, pos, with_labels=True, edge_color='gray', node_size=50, font_size=6)

    #Plot solution
    edges = [(solution[i], solution[i + 1]) for i in range(len(solution) - 1)] + [(solution[-1], solution[0])]
    nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='red', width=2)
    plt.title("TSP Graph with Solution")
    plt.show()

def create_chromosome_solution(graph):
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    return nodes

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

def calculate_fitness(solution, graph):
    total_distance = 0   
    for i in range(len(solution) - 1):
        if solution[i] in graph  and solution[i + 1] in graph[solution[i]]:
            total_distance += graph[solution[i]][solution[i + 1]]['weight']
    # Return to the start
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

def rank_selection(population, fitness_values):
    ranked_indices = list(np.argsort(fitness_values))
    ranks = np.arange(1, len(population) + 1)
    probabilities = ranks / ranks.sum()
    selected_parents_indices = np.random.choice(ranked_indices, size=2, replace=False, p=probabilities)
    return [population[i] for i in selected_parents_indices]

def tournament_selection(population, fitness_values, tournament_size=5):
    selected_parents = []
    for _ in range(2):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        selected_parents.append(population[tournament_indices[np.argmax(tournament_fitness)]])
    return selected_parents

def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fit / total_fitness for fit in fitness_values]
    return random.choices(population, weights=probabilities, k=2)

def plot_evolution(best_distances):
    plt.figure(figsize=(10, 6))
    plt.plot(best_distances, marker='o')
    plt.title('Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()

def genetic_algorithm(graph, population_size, generations,probability, selection_func, crossover_func, mutation_func):

    distances_history = []
    best_distance=0
    best_solution=[]
    
    population = [create_chromosome_solution(graph) for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness
        fitness_values = [calculate_fitness(solution, graph) for solution in population]

        # Crossover
        selected_parents = selection_func(population, fitness_values)

        # Selection
        children = [crossover_func(selected_parents[0], selected_parents[1]) for _ in range(population_size // 2)]
        
        # Mutation
        for i in range(len(children)):
            if random.uniform(0, 1) < probability:  # Mutation probability
                children[i] = mutation_func(children[i])

        temp_distance = min(fitness_values)
        temp_distance_index= fitness_values.index(temp_distance)
        temp_solution = population[temp_distance_index]

        # Replace old population
        population = selected_parents + children  

        if(best_distance==0  or temp_distance< best_distance):
            best_distance=temp_distance
            best_solution = temp_solution

        distances_history.append(temp_distance)

    # Best solution
    return best_solution, best_distance,distances_history

if __name__ == "__main__":
    #TSP file
    problem_file = "datasets/ulysses16.tsp"
    tsp_problem = tsplib95.load(problem_file)

    tsp_graph = create_tsp_graph(tsp_problem)

    population_size = 50
    generations = 100
    probability=0.1

    # GA with different options
    best_solution, best_distance, distances_history = genetic_algorithm(tsp_graph, population_size, generations,probability,
                                                     selection_func=tournament_selection,
                                                     crossover_func=one_point_crossover,
                                                     mutation_func=always_one_mutation)

    print("Best Solution:", best_solution)
    print("Best Distance:", best_distance)

    plot_tsp_graph(tsp_graph,best_solution)

    plot_evolution(distances_history)