from random import randint, sample, uniform, choice
from bisect import bisect_left
from typing import Tuple, List, Dict, Set
from random import random, shuffle


# ===========================
# === Crossover Operators ===
# ===========================


def order_crossover(p1: Tuple[int], p2: Tuple[int]) -> List[Tuple[int]]:
    size = len(p1)
    ch1, ch2 = [-1]*size, [-1]*size
    cx_point1 = randint(0, size - 2)
    cx_point2 = randint(cx_point1 + 1, size - 1)
    ch1[cx_point1:cx_point2 + 1] = p1[cx_point1:cx_point2 + 1]
    ch2[cx_point1:cx_point2 + 1] = p2[cx_point1:cx_point2 + 1]

    def fill(child, parent):
        pos = 0
        used = set(child)
        for i in range(size):
            if child[i] == -1:
                while parent[pos] in used:
                    pos += 1
                child[i] = parent[pos]
                used.add(parent[pos])
        return child

    return [tuple(fill(ch1, p2)), tuple(fill(ch2, p1))]

#Somehow the correct wersion under neath perfoms way worse than incorect
#I thikn it might be because it follows the structure of both parents way too close
'''
def order_crossover(p1: Tuple[int], p2: Tuple[int]) -> List[Tuple[int]]:
    size = len(p1)
    ch1, ch2 = [-1]*size, [-1]*size
    cx_point1 = randint(0, size - 2)
    cx_point2 = randint(cx_point1 + 1, size - 1)
    ch1[cx_point1:cx_point2 + 1] = p1[cx_point1:cx_point2 + 1]
    ch2[cx_point1:cx_point2 + 1] = p2[cx_point1:cx_point2 + 1]

    def fill(child, parent, start_pos):
        pos = (start_pos + 1) % size  # Start AFTER second crossover point
        used = set(child)
        for i in range(size):
            if child[i] == -1:
                while parent[pos] in used:
                    pos = (pos + 1) % size
                child[i] = parent[pos]
                used.add(parent[pos])
        return child

    return [tuple(fill(ch1, p2, cx_point2)), tuple(fill(ch2, p1, cx_point2))]
'''
    


def pmx_crossover(p1: Tuple[int], p2: Tuple[int]) -> List[Tuple[int]]:
    size = len(p1)
    ch1, ch2 = [-1]*size, [-1]*size
    cx_point1 = randint(0, size - 2)
    cx_point2 = randint(cx_point1 + 1, size - 1)
    ch1[cx_point1:cx_point2 + 1] = p1[cx_point1:cx_point2 + 1]
    ch2[cx_point1:cx_point2 + 1] = p2[cx_point1:cx_point2 + 1]

    def mapping(child, p_copied, p_remaining, cx_point1, cx_point2):
        for pos in range(cx_point1, cx_point2 + 1):
            val = p_remaining[pos]
            if val not in child:
                mapped = p_copied[pos]
                target_pos = p_remaining.index(mapped)
                while child[target_pos] != -1:
                    mapped = p_copied[target_pos]
                    target_pos = p_remaining.index(mapped)
                child[target_pos] = val
        return child

    def fill(child, parent):
        pos = 0
        used = set(child)
        for i in range(size):
            if child[i] == -1:
                while parent[pos] in used:
                    pos += 1
                child[i] = parent[pos]
                used.add(parent[pos])
        return child

    return [
        tuple(fill(mapping(ch1, p1, p2, cx_point1, cx_point2), p2)),
        tuple(fill(mapping(ch2, p2, p1, cx_point1, cx_point2), p1))
    ]


def build_edge_table(p1: Tuple[int], p2: Tuple[int]) -> Dict[int, Set[int]]:
    def get_neighbors(tour: Tuple[int], i: int) -> Set[int]:
        idx = tour.index(i)
        n = len(tour)
        return {tour[(idx - 1) % n], tour[(idx + 1) % n]}

    edge_table = {i: set() for i in p1}
    for parent in [p1, p2]:
        for city in parent:
            edge_table[city].update(get_neighbors(parent, city))
    return edge_table

def edge_crossover(p1: Tuple[int], p2: Tuple[int]) -> List[Tuple[int]]:
    n = len(p1)
    edge_table = build_edge_table(p1, p2)
    child = [0] 
    visited = {0}
    current = 0

    while len(child) < n:
        for edges in edge_table.values():
            edges.discard(current)
        
        neighbors = edge_table[current]
        if neighbors:
            next_city = min(neighbors, key=lambda x: len(edge_table[x]))
        else:
            remaining = [c for c in range(1, n) if c not in visited]
            next_city = choice(remaining)
        
        child.append(next_city)
        visited.add(next_city)
        current = next_city

    return [tuple(child)]


def cycle_crossover(parent1: Tuple[int, ...], parent2: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    size = len(parent1)
    offspring1 = [-1] * size
    offspring2 = [-1] * size
    
    # Start the cycle from the first unmatched index
    index = 0
    while -1 in offspring1:
        # If this index is already filled, move to next
        if offspring1[index] != -1:
            index = offspring1.index(-1)
        
        # Start a cycle
        start = index
        val = parent1[index]
        while True:
            offspring1[index] = parent1[index]
            offspring2[index] = parent2[index]
            index = parent1.index(parent2[index])
            if index == start:
                break

        # Fill remaining positions from the opposite parent
        for i in range(size):
            if offspring1[i] == -1:
                offspring1[i] = parent2[i]
                offspring2[i] = parent1[i]
    
    return tuple(offspring1), tuple(offspring2)


# ==========================
# === Mutation Operators ===
# ==========================

def inversion_mutation(ind: Tuple[int]) -> Tuple[int, ...]:
    size = len(ind)
    i = randint(1, size - 2)
    j = randint(i + 1, size - 1)
    new = list(ind)
    new[i:j+1] = reversed(new[i:j+1])
    return tuple(new)

def insert_mutation(ind: Tuple[int]) -> Tuple[int, ...]:
    size = len(ind)
    i = randint(0, size - 2)
    j = randint(i + 1, size - 1)
    new1 = ind[:i + 1]
    new2 = list(ind[i + 1:])
    moved = new2.pop(j - i - 1)
    new = new1 + (moved,) + tuple(new2)
    return new

def swap_mutation(ind: Tuple[int]) -> Tuple[int, ...]:
    size = len(ind)
    i = randint(0, size - 2)
    j = randint(i + 1, size - 1)
    indl = list(ind)
    indl[i], indl[j] = indl[j], indl[i]
    return tuple(indl)

def scramble_mutation(ind: Tuple[int]) -> Tuple[int, ...]:
    head = ind[0]
    tail = list(ind[1:])  
    shuffle(tail)
    mutated = (head,) + tuple(tail)
    return mutated

# ===========================
# === Selection Operators ===
# ===========================

def tournament_selection(fitness_population: List[Tuple[float, Tuple[int]]], selection_params: dict, num_parents: int) -> List[Tuple[float, Tuple[int]]]:
    selected = []
    for _ in range(num_parents):
        group = sample(fitness_population, selection_params.get("k"))
        winner = min(group, key=lambda x: x[0])
        selected.append(winner)
    return selected


def sus_selection(fitness_population: List[Tuple[float, Tuple[int]]], selection_params: dict, num_parents: int) -> List[Tuple[float, Tuple[int]]]:
    fitness_values = [1 / (1 + dist) for dist, _ in fitness_population]
    total = sum(fitness_values)
    probs = [f / total for f in fitness_values]

    cumulative, current = [], 0
    for p in probs:
        current += p
        cumulative.append(current)
    cumulative[-1] = 1.0

    start = uniform(0, 1 / num_parents)
    pointers = [start + i / num_parents for i in range(num_parents)]
    return [fitness_population[bisect_left(cumulative, p)] for p in pointers]


def roulette_wheel_selection(fitness_population: List[Tuple[float, Tuple[int]]], selection_params: dict, num_parents: int) -> List[Tuple[float, Tuple[int]]]:
    fitness_values = [1 / (1 + dist) for dist, _ in fitness_population]  
    total = sum(fitness_values)
    probs = [f / total for f in fitness_values]
    
    cumulative, current = [], 0
    for p in probs:
        current += p
        cumulative.append(current)
    cumulative[-1] = 1.0  

    selected = []
    for _ in range(num_parents):
        idx = bisect_left(cumulative, random())
        selected.append(fitness_population[idx])
    
    return selected


# =========================
# === Operator Registry ===
# =========================

CROSSOVERS = {
    "OX": order_crossover,
    "PMX": pmx_crossover,
    "ERX": edge_crossover,
    "CX": cycle_crossover,
}

MUTATIONS = {
    "inversion": inversion_mutation,
    "insert": insert_mutation,
    "swap": swap_mutation,
    "scramble": scramble_mutation,
}

SELECTIONS = {
    "tournament": tournament_selection,
    "sus": sus_selection,
    "roulette wheel": roulette_wheel_selection,
}

