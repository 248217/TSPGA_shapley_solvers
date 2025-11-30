from typing import List, Tuple
from random import Random
from math import dist
from pandas import DataFrame


class TSPProblem:
    def __init__(self, size: int, seed: int = 42) -> None:
        self.size: int = size
        self.seed: int = seed
        self.coordinates: List[Tuple[float, float]] = []
        self.distance_matrix: List[List[float]] = []
        self.set_of_cities: List[int] = [i for i in range(self.size)]

    def generate_coordinates(self) -> None:
        rng = Random(self.seed)
        self.coordinates = [
            (rng.randint(0, self.size * 2), rng.randint(0, self.size * 2))
            for _ in range(self.size)
        ]
        self._compute_distance_matrix()

    def load_coordinates_from_lists(self, x_coords: List[float], y_coords: List[float]) -> None:
        if len(x_coords) != len(y_coords):
            raise ValueError("x_coords and y_coords must be the same length")
        
        self.coordinates = list(zip(x_coords, y_coords))
        self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> None:
        self.distance_matrix = [
            [dist(self.coordinates[i], self.coordinates[j]) for j in range(self.size)]
            for i in range(self.size)
        ]

    def print_coordinates(self) -> None:
        print("\n" + "=" * 10 + " Coordinates " + "=" * 10)
        print(f"number of coordinates: {len(self.coordinates)}, coordinates: {self.coordinates}")
        print("=" * 33)

    def print_distance_matrix(self) -> None:
        print("\n" + "=" * 10 + " Distance Matrix " + "=" * 10)
        for row in self.distance_matrix:
            for value in row:
                print(f"{value:.2f}", end="  ")
            print() 
        print("=" * 37)

    def extract_coords_to_csv(self, filename: str = "coordinates.csv") -> None:
        if not self.coordinates:
            raise ValueError("No coordinates to save. Generate or load them first.")

        city_names = [f"City_{i}" for i in range(self.size)]
        x_coords = [x for x, _ in self.coordinates]
        y_coords = [y for _, y in self.coordinates]

        df = DataFrame({
            'city': city_names,
            'x': x_coords,
            'y': y_coords
        })

        df.to_csv(filename, index=False)
        print(f"Coordinates saved to {filename}")

    def extract_coords_to_tsp(self, filename: str = "problem.tsp") -> None:
        if not self.coordinates:
            raise ValueError("No coordinates to save. Generate or load them first.")
        
        with open(filename, "w") as f:
            f.write(f"NAME: {self.size}_rnd_gen_{self.seed}\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {self.size}\n")
            f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
            
            for i, (x, y) in enumerate(self.coordinates, 1):
                f.write(f"{i} {x} {y}\n")
            
            f.write("EOF\n")
        
        print(f"TSP file saved to {filename}")
