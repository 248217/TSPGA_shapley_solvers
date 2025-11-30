from TSPsolver.tsp_problem_gen import TSPProblem
from TSPsolver.tspga_config import GAConfig
from TSPsolver.tsp_solver import solve_tsp_ga_serial
from time import time
from random import sample
from math import factorial
from multiprocessing import Pool, cpu_count
from itertools import permutations

def generate_all_permutations(size: int, fixed_first: int = 0):
    base_elements = [i for i in range(size) if i != fixed_first]
    all_perms = [(fixed_first,) + p for p in permutations(base_elements)]
    return all_perms

def generate_samples(size: int, num_samples: int, fixed_first: int = 0):
    base_elements = tuple(i for i in range(size) if i != fixed_first)
    num_samples = min(num_samples, min(factorial(size-1), 10000))
    samples = set() 
    while len(samples) < num_samples:
        perm = (fixed_first,) + tuple(sample(base_elements, len(base_elements)))
        samples.add(perm)
    return samples

def get_or_compute_tsp(problem, config, subset_with, stored_tours, store_limit):
        key = frozenset(subset_with) 
        if key in stored_tours:
            return stored_tours[key]
        tspLen, _, _ = solve_tsp_ga_serial(problem, config, subset_with, False)
        if len(subset_with) >= 9 and len(stored_tours) < store_limit:
            stored_tours[key] = tspLen
        return tspLen

def solve_Shapley(problem: TSPProblem, config: GAConfig, num_samples: int, store_limit: int = 2**14, should_log: bool = True):
    marginalContributions = {i: [] for i in range(1, problem.size)}
    samples = generate_samples(problem.size, num_samples)
    #samples = generate_all_permutations(problem.size) used for benchmark
    stored_tours = {}
    
    grand_set = [i for i in range(problem.size)]
    key = frozenset(grand_set)
    tspLen, tspTour, _ = solve_tsp_ga_serial(problem, config, grand_set, False)
    stored_tours[key] = tspLen
    
    total = len(samples)
    if should_log: print(f"Computing Shapley values from {total} samples")
    for i, sample in enumerate(samples):
        if (i % (total // 100 or 1) == 0) and should_log:
            print(f"\r Progress:: {i / total:.1%}", end='', flush=True)
        cost_without = 0.0
        for i in range(1, problem.size): 
            subset_with = list(sample[:i+1])  
            cost_with = get_or_compute_tsp(problem, config, subset_with, stored_tours, store_limit)
            #cost_with = solve_tsp_ga_serial(problem, config, subset_with, False)
            marginal_contribution = cost_with - cost_without
            cost_without = cost_with
            marginalContributions[sample[i]].append(marginal_contribution)
    if should_log: print("\rProgress: 100.0%")

    shapley_values = {key: round(sum(marginalContributions[key]) / len(samples), 3) for key in marginalContributions}
    return shapley_values, tspLen, tspTour
    
def solve_Shapley_prl(problem: TSPProblem, config: GAConfig, num_samples: int, store_limit: int = 2**14):
    num_processes = max(1, cpu_count() - 1)
    with Pool(processes=num_processes) as pool:
        args = [(problem, config, num_samples, store_limit, i==num_processes-1) for i in range(num_processes)]
        results = pool.starmap(solve_Shapley, args)

    extract_by_key = {i: [] for i in range(1, problem.size)}
    extract_lengths = []
    for i, result in enumerate(results):
        sumShap = 0
        for key in result[0]:
            sumShap += result[0][key]
            extract_by_key[key].append(round(result[0][key],3))
        extract_lengths.append(result[1])
        print(f"=== Thread {i+1} ===")
        print(f"Shapley values for each thread are: {result[0]}")
        print(f"TSP length for all elements: {result[1]:.3f} sum of shapley values {sumShap:.3f}")

    combined_shapley_values = {key: round(sum(extract_by_key[key]) / num_processes, 3) for key in extract_by_key}
    
    average_grand_tour_length = round(sum(extract_lengths) / num_processes, 3)
    min_index, shortest_grand_tour_length = min(enumerate(extract_lengths), key=lambda x: x[1])
    adjusted_shapley_values =  {key: round((combined_shapley_values[key] / average_grand_tour_length)*shortest_grand_tour_length, 3) for key in combined_shapley_values}
   
    shortest_grand_tour = results[min_index][2]

    return adjusted_shapley_values, shortest_grand_tour_length, shortest_grand_tour

def shapley_interface():
    print(" === Shapley value solver === ")
    size = min(max(3, int(input("Enter number of cities (3-30): "))), 30)
    compute_prl = ("y" == input("do you want to compute Shapley values in parallel? (y): "))
    print(f"Parrallel computing on: {compute_prl}")
    num_samples = max(1, int(input("Choose number of samples per thread >0: ")))
    print(f" problem size: {size}, number of samples: {num_samples}")

    # Default configuration
    cfg = {
        "size": size,
        "max_gen": 1000,
        "no_improvement_limit": 20,
        "mode": "linear",
        "size_parameter": 10,
        "max_pop": 10000,
        "selection_ratio": 0.5,
        "mutation_rate": 0.2,
        "crossover_op": "OX",
        "mutation_op": "inversion",
        "selection_op": "tournament",
        "tournament_k": 3,
        "surv_strat": "elitism",
        "elitism_rate": 0.05,
    }
      
    problem = TSPProblem(cfg["size"])
    problem.generate_coordinates()
    config = GAConfig(problem_size=cfg["size"])

    start = time()
    if compute_prl:
        shapley_values, tspSol, tour = solve_Shapley_prl(problem, config, num_samples)
    else:
        shapley_values, tspSol, tour = solve_Shapley(problem, config, num_samples)
    end = time()
    sumShap = 0
    for i in shapley_values:
        sumShap += shapley_values[i]
    print(f"=== Solution ===")
    print(f"Shapley values for given problem are: {shapley_values}")
    print(f"TSP length for all elements: {tspSol:.3f} sum of shapley values {sumShap:.3f}")
    print(f"Grand tour: {tour}")
    print(f"Elapsed time: {end - start:.3f} s")
    print(f"time per sample: {(end - start)/num_samples:.5f}s")

if __name__ == "__main__":
    shapley_interface()

    