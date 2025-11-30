from dataclasses import dataclass, field
from typing import Dict, Any
from math import factorial


@dataclass
class PopulationConfig:
    size_mode: str = "linear"
    size_parameter: float = 10.0
    max_population: int = 10000

    def get_population_size(self, problem_size: int) -> int:
        fact_limit = factorial(problem_size) - 1
        if self.size_mode == "constant":
            return min(min(int(self.size_parameter), fact_limit), self.max_population)
        elif self.size_mode == "linear":
            return min(min(int(self.size_parameter * problem_size), fact_limit), self.max_population)
        elif self.size_mode == "power":
            return min(min(int(problem_size ** self.size_parameter), fact_limit), self.max_population)
        elif self.size_mode == "factorial":
            return min(fact_limit, self.max_population)
        raise ValueError(f"Unknown population size mode: {self.size_mode}")


@dataclass
class OperatorConfig:
    selection: str = "tournament"
    crossover: str = "OX"
    mutation: str = "inversion"
    selection_ratio: float = 0.7
    mutation_rate: float = 0.2
    survivor_strategy: str = "elitism"

    VALID_SELECTION = {"tournament", "sus", "roulette wheel"}
    VALID_CROSSOVER = {"OX", "PMX", "ERX", "CX"}
    VALID_MUTATION = {"inversion", "insert", "swap", "scramble"}
    VALID_SURVIVORS = {"elitism"}

    selection_params: Dict[str, Any] = field(default_factory=dict)
    survivor_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.selection not in self.VALID_SELECTION:
            raise ValueError(f"Invalid selection operator: {self.selection}")
        if self.crossover not in self.VALID_CROSSOVER:
            raise ValueError(f"Invalid crossover operator: {self.crossover}")
        if self.mutation not in self.VALID_MUTATION:
            raise ValueError(f"Invalid mutation operator: {self.mutation}")
        if self.survivor_strategy not in self.VALID_SURVIVORS:
            raise ValueError(f"Invalid survivor strategy: {self.survivor_strategy}")

        if self.selection == "tournament":
            self.selection_params["k"] = self.selection_params.get("k", 3)
        if self.survivor_strategy == "elitism":
            self.survivor_params["e"] = self.survivor_params.get("e", 0.1)


@dataclass
class TerminationConfig:
    max_generations: int = 1000
    no_improvement_limit: int = 20


class GAConfig:
    def __init__(self, problem_size: int):
        self.population = PopulationConfig()
        self.operators = OperatorConfig()
        self.termination = TerminationConfig()
        self.population_size = self.population.get_population_size(problem_size)

    def print_config(self):
        print("\n" + "=" * 10 + " GA Configuration " + "=" * 10)
        print(f"Population size mode: {self.population.size_mode}")
        print(f"  size parameter: {self.population.size_parameter}")
        print(f"  population size for given parameters: {self.population_size}")
        print(f"  max allowed population: {self.population.max_population}")
        print(f"Selection method: {self.operators.selection}")
        print(f"  Tournament size: {self.operators.selection_params.get('k', 'N/A')}")
        print(f"Selection ratio: {self.operators.selection_ratio}")
        print(f"Crossover: {self.operators.crossover}")
        print(f"Mutation: {self.operators.mutation} (mutation rate={self.operators.mutation_rate})")
        print(f"Survivor strategy: {self.operators.survivor_strategy}")
        print(f"  Elitism rate: {self.operators.survivor_params.get('e', 'N/A')}")
        print(f"Max generations: {self.termination.max_generations}")
        print(f"No improvement limit: {self.termination.no_improvement_limit}")
        print("=" * 38)

    def change_configuration(self, cfg: Dict[str, Any], problem_size: int, printCfg = True):
        self.population.size_mode = cfg["mode"]
        self.population.size_parameter = cfg["size_parameter"]
        self.population.max_population = cfg["max_pop"]
        self.population_size = self.population.get_population_size(problem_size)

        self.operators.selection_ratio = cfg["selection_ratio"]
        self.operators.selection = cfg["selection_op"]
        self.operators.selection_params = {}
        if cfg["selection_op"] == "tournament":
            self.operators.selection_params["k"] = cfg["tournament_k"]

        self.operators.crossover = cfg["crossover_op"]

        self.operators.mutation = cfg["mutation_op"]
        self.operators.mutation_rate = cfg["mutation_rate"]

        self.operators.survivor_strategy = cfg["surv_strat"]
        self.operators.survivor_params = {"e": cfg["elitism_rate"]}

        self.termination.max_generations = cfg["max_gen"]
        self.termination.no_improvement_limit = cfg["no_improvement_limit"]
        
        if printCfg:
            self.print_config()

