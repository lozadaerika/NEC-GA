import tsplib95
import random
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

def plot_tsp_graph(graph, solution,image_name):
    #Plot graph
    pos = nx.get_node_attributes(graph, 'coords')
    nx.draw(graph, pos, with_labels=True, edge_color='gray', node_size=50, font_size=6)

    #Plot solution
    edges = [(solution[i], solution[i + 1]) for i in range(len(solution) - 1)] + [(solution[-1], solution[0])]
    nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='red', width=2)
    plt.title("TSP Graph with Solution")
    plt.savefig(image_name)
    plt.clf()
    #plt.show()

def plot_evolution(best_distances,image_name):
    
    plt.figure(figsize=(10, 6))
    plt.plot(best_distances, marker='o')
    plt.title('Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.savefig(image_name)
    plt.clf()
    #plt.show()

def calculate_euclidean_distance(coord1, coord2):
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5

def create_chromosome_solution(graph):
    chromosome = list(graph.nodes())
    random.shuffle(chromosome)
    return chromosome

def calculate_fitness(individual, graph):
    total_distance = 0   
    for i in range(len(individual) - 1):
        if individual[i] in graph  and individual[i + 1] in graph[individual[i]]:
            total_distance += graph[individual[i]][individual[i + 1]]['weight']
    return total_distance

def inversion_mutation(chromosome,mutation_rate):
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(len(chromosome)), 2))
        chromosome[start:end + 1] = reversed(chromosome[start:end + 1])
    return chromosome

def scramble_mutation(original_chomosome, mutation_rate):
    chromosome = original_chomosome.copy()
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            start_index = random.randint(0, len(chromosome) - 1)
            end_index = random.randint(start_index, len(chromosome) - 1)
            segment = chromosome[start_index:end_index + 1]
            random.shuffle(segment)
            chromosome[start_index:end_index + 1] = segment
    return chromosome

def swap_mutation(chromosome,mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        temp1 = chromosome[idx1]
        temp2 = chromosome[idx2]
        chromosome[idx1] = temp2
        chromosome[idx2] = temp1
    return chromosome

def partial_mapped_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    mapping1 = {parent1[i]: parent2[i] for i in range(start, end)}
    mapping2 = {parent2[i]: parent1[i] for i in range(start, end)}
    child1 = [mapping1.get(city, city) for city in parent1]
    child2 = [mapping2.get(city, city) for city in parent2]
    child1=legalize_mapping_offstring(start,end,mapping2,child1)
    child2=legalize_mapping_offstring(start,end,mapping1,child2)
    return [child1, child2]

def legalize_mapping_offstring(start, end,mapping,child):
    while len(child) != len(set(child)):
        for i in range(len(child)):
            if(i<start or i>=end):
                if child[i] in mapping:
                    child[i]=mapping.get(child[i])
    return child

def ordered_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child1 = [-1] * len(parent1)
    child2 = child1.copy()
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
    child1=complete_order_crossover(parent2,child1)
    child2=complete_order_crossover(parent1,child2)
    return [child1, child2]

def complete_order_crossover(parent, child):
    index = 0
    for i in range(len(parent)):
        if child[i] == -1:
            while parent[index] in child:
                index += 1
            child[i] = parent[index]
    return child

def rank_selection(population,problem, fitness_values):
    fitnesses = []
    for i in range(len(population)):
        fitnesses.append({'id':i,'fitness':calculate_fitness(population[i],problem),'problem':population[i]})
    sorted_descriptions = [entry['problem'] for entry in sorted(fitnesses, key=lambda x: x['fitness'])]
    return sorted_descriptions

def tournament_selection(population, problem,fitness_values):
    selected_parents = []
    for _ in range(len(population)):
        tournament_indices = random.sample(population,fitness_values)
        selected_parents.append(min(tournament_indices, key=lambda x: calculate_fitness(x, problem)))
    return selected_parents

def genetic_algorithm(graph, population_size, generations,mutation_rate, selection_func, crossover_func, mutation_func):

    distances_history = []
    population = [create_chromosome_solution(graph) for _ in range(population_size)]

    for generation in range(generations):
        new_population=[]
        # Selection
        selected_parents = selection_func(population, graph,5 )     
        for i in range(len(selected_parents) // 2):
            # Crossover
            children = crossover_func(selected_parents[i], selected_parents[i + 1])
            # Mutation
            for i in range(len(children)):
                children[i] = mutation_func(children[i],mutation_rate)
            new_population.extend([children[0], children[1]])
        # Replace old population   
        population = new_population
        # Best solution
        best_solution = min(population, key=lambda x: calculate_fitness(x, graph))
        best_distance = calculate_fitness(best_solution, graph)
        distances_history.append(best_distance)
    return best_solution, best_distance,distances_history

if __name__ == "__main__":

    population_sizes=[50,100,200,300,500]
    generation_sizes=[10,30,50,100]
    mutation_rates=[0.6,0.8]

    selection_functions=[rank_selection,tournament_selection]
    mutation_functions=[swap_mutation,inversion_mutation]
    crossover_functions=[ordered_crossover,partial_mapped_crossover]

    datasets=['ulysses16.tsp','ulysses22.tsp','att48.tsp','berlin52.tsp','st70.tsp']

    for dataset in datasets:

        #TSP file
        problem_file = dataset
        tsp_problem = tsplib95.load('datasets/'+problem_file)

        tsp_graph = create_tsp_graph(tsp_problem)

        for selection_fn in selection_functions:
            for mutation_fn in mutation_functions:
                for crossover_fn in crossover_functions:

                    best_population=0
                    best_generation=0
                    best_mutation=0
                    best_distance=0
                    best_solution=[]
                    distances_history=[]

                    for population_size in population_sizes:
                        for generations in generation_sizes:
                            for mutation_rate in mutation_rates:
                                print('Population size',population_size,'Generations',generations,'Mutation rate',mutation_rate)
                                # GA with different options
                                temp_best_solution, temp_best_distance, distances_history = genetic_algorithm(tsp_graph, population_size, generations,mutation_rate,
                                                                                selection_func=selection_fn,
                                                                                crossover_func=crossover_fn,
                                                                                mutation_func=mutation_fn)
                                
                                if(best_distance==0  or temp_best_distance< best_distance):
                                    best_distance=temp_best_distance
                                    best_solution= temp_best_solution     
                                    best_population=population_size
                                    best_generation=generations 
                                    best_mutation=mutation_rate      

                    filename=problem_file +"-"+ str(selection_fn.__name__) +"-"+  str(mutation_fn.__name__) +"-"+ str(crossover_fn.__name__) + "-"+str(best_generation)+  "-"+str(best_population) +"-"+str(best_mutation)               
                 
                    with open("images/output-"+filename+".txt", 'w') as file:      
                        sys.stdout = file  # Redirect stdout
                        print(problem_file,'Selection:',selection_fn.__name__,'Crossover:',crossover_fn.__name__)
                        print(problem_file,'Mutation:',mutation_fn.__name__,'Mutation rate:',best_mutation)
                        print(problem_file,"Best Parameters: ", 'Population size',best_population,'Generations',best_generation)
                        print(problem_file,"Best Solution:", best_solution)
                        print(problem_file,"Best Distance:", best_distance)

                    # Reset stdout to the original value
                    sys.stdout = sys.__stdout__

                    print(problem_file,'Selection:',selection_fn.__name__,'Crossover:',crossover_fn.__name__)
                    print(problem_file,'Mutation:',mutation_fn.__name__,'Mutation rate:',best_mutation)
                    print(problem_file,"Best Parameters: ", 'Population size',best_population,'Generations',best_generation)
                    print(problem_file,"Best Solution:", best_solution)
                    print(problem_file,"Best Distance:", best_distance)
                    
                    plot_tsp_graph(tsp_graph,best_solution,"images/graph-"+filename+".png")
                    plot_evolution(distances_history,"images/evol-"+filename+".png")