from itertools import permutations
from typing import Tuple, List
from .tsp_problem_gen import TSPProblem
from .tspga_config import GAConfig
from .pop_manager import (
    initialize_population,
    evaluate_fitness,
    select_parents,
    crossover,
    mutate,
    select_survivors,
    evaluate_fitness_prl,
    crossover_prl,
)
from multiprocessing import Pool, cpu_count
from time import time


def should_terminate(generation: int, no_improvement: int, max_generations: int, no_improvement_limit: int) -> bool:
    return generation >= max_generations or no_improvement >= no_improvement_limit

def solve_exactly_with_permutations(problem: TSPProblem, set_of_cities: List[int], should_log: bool) -> Tuple[Tuple[float, Tuple[int]], List[float], List[Tuple[int, int]]]:
    if should_log:
        print("Small problem detected, solving exactly with all permutations...")
    fixed_start = set_of_cities[0]
    others = set_of_cities[1:]
    population = [(fixed_start,) + perm for perm in permutations(others)]
    fitness_population = evaluate_fitness(population, problem.coordinates)
    best = min(fitness_population, key=lambda x: x[0])
    best_history = [best[0]]
    return best, best_history


def solve_tsp_ga_serial(problem: TSPProblem, config: GAConfig, set_of_cities: List[int], should_log: bool = True) -> Tuple[float, Tuple[int], List[float]]:
    if len(set_of_cities) < 9:
        if  len(set_of_cities) <= 1: return 0, (0,), [0]
        best, best_history = solve_exactly_with_permutations(problem, set_of_cities, should_log)
        return best[0], best[1], best_history
    
    population = initialize_population(set_of_cities, config.population_size)
    fitness_population = evaluate_fitness(population, problem.coordinates)

    generation: int = 0
    no_improvement: int = 0
    best_history: List[float] = []
    best_overall: Tuple[float, Tuple[int]] = (float('inf'), ())

    while not (generation >= config.termination.max_generations or no_improvement >= config.termination.no_improvement_limit):
        fitness_parents = select_parents(fitness_population, config.operators.selection_ratio, config.operators.selection, config.operators.selection_params)
        children = crossover(fitness_parents, config.operators.crossover, config.operators.survivor_params, config.population_size)
        children = mutate(children, config.operators.mutation, config.operators.mutation_rate)
        fitness_children = evaluate_fitness(children, problem.coordinates)
        survivors = select_survivors(fitness_population, config.operators.survivor_params, config.population_size)
        fitness_population = survivors + fitness_children

        current_best = survivors[0]
        best_history.append(current_best[0])
        if current_best[0] < best_overall[0]:
            best_overall = current_best
            no_improvement = 0

        if should_log:
            print(f"Generation {generation}: best length = {current_best[0]:.2f}, no improvement for {no_improvement} generations.")
        generation += 1
        no_improvement += 1

    return best_overall[0], best_overall[1], best_history

def solve_tsp_ga_parallel(problem: TSPProblem, config: GAConfig, set_of_cities: List[int], should_log: bool = True) -> Tuple[float, Tuple[int], List[float]]:
    if  len(set_of_cities) < 9:
        if  len(set_of_cities) <= 1: return 0, (0,), [0]
        best, best_history = solve_exactly_with_permutations(problem, set_of_cities, should_log)
        return best[0], best[1], best_history

    num_processes = cpu_count() -1
    with Pool(processes=num_processes) as pool:
        population = initialize_population(set_of_cities, config.population_size)
        fitness_population = evaluate_fitness_prl(population, problem.coordinates, num_processes, pool)

        generation: int = 0
        no_improvement: int = 0
        best_history: List[float] = []
        best_overall: Tuple[float, Tuple[int]] = (float('inf'), ())

        while not (generation >= config.termination.max_generations or no_improvement >= config.termination.no_improvement_limit):
            #fitness_parents = select_parents_prl(fitness_population, config.operators.selection, config.operators.selection_params, num_processes, pool)
            fitness_parents = select_parents(fitness_population, config.operators.selection_ratio, config.operators.selection, config.operators.selection_params)
            children = crossover_prl(fitness_parents, config.operators.crossover, config.operators.survivor_params, config.population_size, num_processes, pool)
            #children = mutate_prl(children, config.operators.mutation, config.operators.mutation_rate, num_processes, pool)
            children = mutate(children, config.operators.mutation, config.operators.mutation_rate)
            fitness_children = evaluate_fitness_prl(children, problem.coordinates, num_processes, pool)
            survivors = select_survivors(fitness_population, config.operators.survivor_params, config.population_size)
            fitness_population = survivors + fitness_children

            current_best = survivors[0]
            best_history.append(current_best[0])
            if current_best[0] < best_overall[0]:
                best_overall = current_best
                no_improvement = 0
            if should_log:
                print(f"Generation {generation}: best length = {current_best[0]:.2f}, no improvement for {no_improvement} generations.")
            generation += 1
            no_improvement += 1

        return best_overall[0], best_overall[1], best_history

