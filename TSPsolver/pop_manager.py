from random import sample, random
from math import ceil, dist 
# import multiprocessing as mp
from typing import List, Tuple
from .operators import CROSSOVERS, MUTATIONS, SELECTIONS

# --- Nonparallel utilities --- 
def initialize_population(set_of_cities: Tuple[int], population_size: int) -> List[Tuple[int]]:
    fixed_start = 0
    base = [i for i in set_of_cities if i != fixed_start]
    population: set[tuple[int, ...]] = set()
    while len(population) < population_size:
        individual = (fixed_start,) + tuple(sample(base, len(base)))
        population.add(individual)
    return list(population)


def evaluate_fitness(population: List[Tuple[int]], coordinates: List[Tuple[float, float]]) -> List[Tuple[float, Tuple[int]]]:
    fitness_population = []
    for tour in population:
        distance = 0
        for p1, p2 in zip(tour, tour[1:] + tour[:1]):
            distance += dist(coordinates[p1], coordinates[p2])
        fitness_population.append((distance, tour))
    return fitness_population

'''
def distance_of_tour(tour: Tuple[int], distance_matrix: List[List[float]]) -> float:
        return sum(
            distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
            for i in range(len(tour))
        )

def evaluate_fitness_dm(population: List[Tuple[int]], distance_matrix: List[List[float]]) -> List[Tuple[float, Tuple[int]]]:
    print("evalueating fitness by DM")
    return [(distance_of_tour(tour, distance_matrix), tour) for tour in population]
'''

def select_parents(fitness_population: List[Tuple[float, Tuple[int]]], selection_ratio: float, selection_operator: str, selection_params: dict) -> List[Tuple[float, Tuple[int]]]:
    num_parents = int((len(fitness_population) + 1)*selection_ratio)
    selector = SELECTIONS[selection_operator]
    return selector(fitness_population, selection_params, num_parents)

def crossover(parents: List[Tuple[float, Tuple[int]]], crossover_operator: str, survivor_params: dict, population_size: int) -> List[Tuple[int]]:
    crossover_func = CROSSOVERS[crossover_operator]
    parent_tours = [tour for _, tour in parents]
    children = []
    e = survivor_params.get("e", 0.1)
    target_children = population_size - int(e * population_size)
    for i in range(0, len(parent_tours) - 1, 2):
        offspring = crossover_func(parent_tours[i], parent_tours[i + 1])
        children += offspring
        if len(children) >= target_children:
            break
    while len(children) < target_children:
        p1, p2 = sample(parent_tours, 2)
        children += crossover_func(p1, p2)
    return children[:target_children]

def mutate(children: List[Tuple[int]], mutation_operator: str, mutation_rate: float) -> List[Tuple[int]]:
    mutation_func = MUTATIONS[mutation_operator]
    return [mutation_func(ind) if random() < mutation_rate else ind for ind in children]

def select_survivors(fitness_population: List[Tuple[float, Tuple[int]]], survivor_params: dict, population_size: int) -> List[Tuple[float, Tuple[int]]]:
    e = survivor_params.get("e", 0.0)
    k = int(e * population_size)
    sorted_population = sorted(fitness_population, key=lambda x: x[0])
    return sorted_population[:k]
    
# --- chunkify ---   
def dynamic_chunkify(lst: list, num_chunks: int) -> List[list]:
    k, m = divmod(len(lst), num_chunks)
    chunks = []
    start = 0
    for i in range(num_chunks):
        chunk_size = k + 1 if i < m else k
        chunks.append(lst[start:start + chunk_size])
        start += chunk_size
    return chunks

# --- Parallel methods ---

def evaluate_fitness_prl(population: List[Tuple[int]], coordinates: List[Tuple[float, float]], num_processes: int, pool) -> List[Tuple[float, Tuple[int]]]:
    chunks = dynamic_chunkify(population, num_processes)
    args = [(chunk, coordinates) for chunk in chunks]
    results = pool.starmap(evaluate_fitness, args)
    return [item for sublist in results for item in sublist]

'''
def evaluate_fitness_prl_dm(population: List[Tuple[int]], dm: List[Tuple[int, int]], num_processes: int, pool) -> List[Tuple[float, Tuple[int]]]:
    print("Parallel DM finteness")
    chunks = dynamic_chunkify(population, num_processes)
    args = [(chunk, dm) for chunk in chunks]
    results = pool.starmap(evaluate_fitness_dm, args)
    return [item for sublist in results for item in sublist]

'''
    
def select_parents_prl(fitness_population: List[Tuple[float, Tuple[int]]], selection_ratio: float, selection_operator: str, selection_params: dict, num_processes: int, pool) -> List[Tuple[float, Tuple[int]]]:
    chunks = dynamic_chunkify(fitness_population, num_processes)
    args = [(chunk, selection_ratio, selection_operator, selection_params) for chunk in chunks]
    results = pool.starmap(select_parents, args)
    return [item for sublist in results for item in sublist]

def crossover_prl(parents: List[Tuple[float, Tuple[int]]], crossover_operator: str, survivor_params: dict, population_size: int, num_processes: int, pool) -> List[Tuple[int]]:
    chunks = dynamic_chunkify(parents, num_processes)
    chunk_pop_size = ceil(population_size / num_processes)
    args = [(chunk, crossover_operator, survivor_params, chunk_pop_size) for chunk in chunks]
    results = pool.starmap(crossover, args)
    return [item for sublist in results for item in sublist]

def mutate_prl(children: List[Tuple[int]], mutation_operator: str, mutation_rate: float, num_processes: int, pool) -> List[Tuple[int]]:
    chunks = dynamic_chunkify(children, num_processes)
    args = [(chunk, mutation_operator, mutation_rate) for chunk in chunks]
    results = pool.starmap(mutate, args)
    return [item for sublist in results for item in sublist]

