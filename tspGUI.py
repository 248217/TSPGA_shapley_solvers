import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import subplots
from matplotlib import style
from threading import Thread
from time import time
from TSPsolver.tsp_problem_gen import TSPProblem
from TSPsolver.tspga_config import GAConfig
from TSPsolver.tsp_solver import solve_tsp_ga_serial, solve_tsp_ga_parallel
from tkinter import filedialog
from pandas import read_csv
from os import path, makedirs, getcwd


class TSPSolverApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TSP Solver - Genetic Algorithm")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("green")

        style.use("dark_background")  # Apply dark matplotlib theme

        self.config_frame = self.build_config_panel()
        self.control_frame = self.build_control_panel()
        self.plot_frame = self.build_plot_panel()
        self.terminal_frame = self.build_terminal()

        self.coordinates = []
        self.tour_line = None
        self.history_line = None
        self.xdata, self.ydata = [], []

        self.problem = None
        self.config = None

        self.redirect_stdout()

    def build_config_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(side="left", fill="y", padx=10, pady=10)

        self.entries = {}
        config_params = [
            ("Cities", 50), ("Generations", 1000), ("No Improvement Limit", 20),
            ("Mutation Rate", 0.2), ("Elitism Rate", 0.1), ("Selection Ratio", 0.7), ("Size Parameter", 10)
        ]

        self.param_bounds = {
            "Cities": (1, 226),
            "Generations": (1, 2000),
            "No Improvement Limit": (0, 60),
            "Mutation Rate": (0.0, 1.0),
            "Elitism Rate": (0.0, 1.0),
            "Selection Ratio": (0.0, 1.0),
            "Size Parameter": (1, 10000),
        }

        for label, default in config_params:
            ctk.CTkLabel(frame, text=label).pack()
            entry = ctk.CTkEntry(frame)
            entry.insert(0, str(default))
            entry.pack()
            self.entries[label] = entry

        self.combo_ops = {}
        dropdowns = {
            "Population Mode": ["linear", "constant", "power", "factorial"],
            "Crossover": ["OX", "PMX", "ERX", "CX"],
            "Mutation": ["inversion", "insert", "swap", "scramble"],
            "Selection": ["tournament", "sus", "roulette wheel"]
        }
        for label, options in dropdowns.items():
            ctk.CTkLabel(frame, text=label).pack()
            combo = ctk.CTkComboBox(frame, values=options, command=self.selection_changed)
            combo.set(options[0])
            combo.pack()
            self.combo_ops[label] = combo

        self.tournament_label = ctk.CTkLabel(frame, text="Tournament Size")
        self.tournament_entry = ctk.CTkEntry(frame)
        self.tournament_entry.insert(0, "3")
        self.tournament_label.pack_forget()
        self.tournament_entry.pack_forget()

        return frame

    def build_control_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(side="top", fill="x", padx=10, pady=5)

        #ctk.CTkButton(frame, text="Save Config", command=self.save_configuration).pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Random Problem", command=self.generate_problem).pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Load Problem", command=self.load_problem_from_csv).pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Start Serial", command=lambda: self.run_solver(mode='serial')).pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Start Parallel", command=lambda: self.run_solver(mode='parallel')).pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Clear Log", command=self.clear_log).pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Export coordinates", command=self.export_data).pack(side="left", padx=5)

        return frame

    def build_plot_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        left_plot_frame = ctk.CTkFrame(frame, corner_radius=20, fg_color="#1e1e1e")
        left_plot_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        right_plot_frame = ctk.CTkFrame(frame, corner_radius=20, fg_color="#1e1e1e")
        right_plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig1, self.ax1 = subplots(figsize=(5, 4))
        self.fig1.patch.set_facecolor('#1e1e1e')
        self.ax1.set_facecolor('#1e1e1e')
        self.ax1.set_title("Fitness over Generations")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Distance")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=left_plot_frame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(fill="both", expand=True)
        self.canvas1.get_tk_widget().configure(bg='#1e1e1e', highlightthickness=0, bd=0)

        self.fig2, self.ax2 = subplots(figsize=(5, 4))
        self.fig2.patch.set_facecolor('#1e1e1e')
        self.ax2.set_facecolor('#1e1e1e')
        self.ax2.set_title("Best Tour")
        self.ax2.set_xlabel("X")
        self.ax2.set_ylabel("Y")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=right_plot_frame)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)
        self.canvas2.get_tk_widget().configure(bg='#1e1e1e', highlightthickness=0, bd=0)

        return frame

    def plot_cities(self, tour: tuple[int] = ()):  # uses self.ax2
        self.ax2.clear()
        self.ax2.set_title("Best tour found")
        self.ax2.set_xlabel("X")
        self.ax2.set_ylabel("Y")

        coords = self.problem.coordinates
        x_all = [c[0] for c in coords]
        y_all = [c[1] for c in coords]

        self.ax2.scatter(x_all[1:], y_all[1:], c='#00FFA3', edgecolors='green')
        self.ax2.scatter(x_all[0], y_all[0], c='#FF69B4', zorder=5)

        for i, (cx, cy) in enumerate(coords):
            self.ax2.text(cx, cy, str(i), fontsize=8, ha='right', va='bottom')

        if tour and len(tour) >= 2:
            for i in range(len(tour) - 1):
                c1 = coords[tour[i]]
                c2 = coords[tour[i + 1]]
                self.ax2.plot([c1[0], c2[0]], [c1[1], c2[1]], color='#00BFFF', linewidth=1.5)

        self.fig2.tight_layout()
        self.canvas2.draw()

    def plot_fitness_history(self, ax, history: list[float] = None):
        if history is None or not history:
            self.log("Warning: No fitness history to plot.")
            return

        ax.clear()
        ax.plot(range(len(history)), history, marker='o', markersize=3, linewidth=2, color='#00FFA3')
        ax.set_title("Best Distance Over Generations")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Distance")
        ax.grid(True, linestyle='--', alpha=0.3)
        self.fig1.tight_layout()
        self.canvas1.draw()

    def selection_changed(self, selection_method):
        if selection_method == "tournament":
            self.tournament_label.pack(pady=2)
            self.tournament_entry.pack(pady=2)
            if not self.tournament_entry.get():
                self.tournament_entry.insert(0, "3")
        else:
            self.tournament_label.pack_forget()
            self.tournament_entry.pack_forget()

    def build_terminal(self):
        frame = ctk.CTkFrame(self)
        frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.terminal = ctk.CTkTextbox(frame, height=150)
        self.terminal.pack(fill="both", expand=True)

        return frame

    def generate_problem(self):
        self.save_configuration()
        size = int(self.entries["Cities"].get())
        self.problem = TSPProblem(size)
        self.problem.generate_coordinates()
        self.log(f"Generated problem with {size} cities.")
        self.plot_cities()

    def clip_and_correct_entry(self, label: str, entry: ctk.CTkEntry):
        bounds = self.param_bounds.get(label)
        val = entry.get()

        if isinstance(bounds, tuple):  # numerical
            try:
                num = float(val)
                clipped = max(bounds[0], min(bounds[1], num))
                corrected = int(clipped) if clipped.is_integer() else clipped
                if num != clipped:
                    self.log(f"{label} value {num} out of bounds; adjusted to {corrected}")
                entry.delete(0, "end")
                entry.insert(0, str(corrected))
                return corrected
            except ValueError:
                self.log(f"{label} is not a valid number; using lower bound {bounds[0]}")
                entry.delete(0, "end")
                entry.insert(0, str(bounds[0]))
                return bounds[0]

        elif isinstance(bounds, list):  # categorical
            if val not in bounds:
                self.log(f"{label} value '{val}' invalid; set to '{bounds[0]}'")
                entry.delete(0, "end")
                entry.insert(0, bounds[0])
                return bounds[0]
            return val

        return val  # If no bounds are defined

    def save_configuration(self):
        clipped_values = {}

        for label, entry in self.entries.items():
            clipped_value = self.clip_and_correct_entry(label, entry)
            clipped_values[label] = clipped_value

        cfg = {
            "size": int(clipped_values["Cities"]),
            "max_gen": int(clipped_values["Generations"]),
            "no_improvement_limit": int(clipped_values["No Improvement Limit"]),
            "mode": self.combo_ops["Population Mode"].get(),
            "size_parameter": float(clipped_values["Size Parameter"]),
            "max_pop": 10000,
            "selection_ratio": float(clipped_values["Selection Ratio"]),
            "mutation_rate": float(clipped_values["Mutation Rate"]),
            "crossover_op": self.combo_ops["Crossover"].get(),
            "mutation_op": self.combo_ops["Mutation"].get(),
            "selection_op": self.combo_ops["Selection"].get(),
            "tournament_k": int(self.tournament_entry.get()) if self.combo_ops["Selection"].get() == "tournament" else None,
            "surv_strat": "elitism",
            "elitism_rate": float(clipped_values["Elitism Rate"]),
        }

        self.config = GAConfig(problem_size=cfg["size"])
        self.config.change_configuration(cfg, cfg["size"])
        self.log("Configuration saved (with bounds enforced).")

    def run_solver(self, mode):
        if not self.problem:
            self.log("Please generate a problem first.")
            return
        self.save_configuration()
        Thread(target=self.solver_thread, args=(mode,), daemon=True).start()

    def solver_thread(self, mode):
        self.clear_log()
        self.xdata.clear()
        self.ydata.clear()
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title("Fitness over Generations")
        self.ax2.set_title("Best Tour")

        start = time()
        if mode == 'serial':
            best_dist, best_tour, history = solve_tsp_ga_serial(self.problem, self.config, self.problem.set_of_cities)
        else:
            best_dist, best_tour, history = solve_tsp_ga_parallel(self.problem, self.config, self.problem.set_of_cities)
        end = time()
        
        self.log("\n=== Computation Completed ===")
        self.log(f"Elapsed time: {end - start:.2f} seconds")
        self.log(f"Time per generation: {(end - start)/len(history):.5f} s")
        self.log(f"Best distance: {best_dist:.2f}")
        self.log(f"Best tour: {best_tour}")

        self.plot_fitness_history(self.ax1, history)
        self.plot_cities(best_tour + (0,))

    def load_problem_from_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            self.log("No file selected.")
            return
        try:
            df = read_csv(file_path)
            if not {'x', 'y'}.issubset(df.columns):
                self.log("CSV must have 'x' and 'y' columns.")
                return
            if len(df['x']) > 226:
                self.log("Problem is too big! Suported size is (1-200)!")
                return

            x_coords = df['x'].tolist()
            y_coords = df['y'].tolist()

            self.problem = TSPProblem(len(x_coords))
            self.problem.load_coordinates_from_lists(x_coords, y_coords)

            self.entries["Cities"].delete(0, "end")
            self.entries["Cities"].insert(0, str(len(x_coords)))

            self.save_configuration()
            self.log(f"Loaded problem from {file_path} with {len(x_coords)} cities.")
            self.plot_cities()

        except Exception as error:
            self.log(f"Error loading CSV: {error}")
    
    def export_data(self):
        def submit_filename():
            filename = entry.get().strip()

            if filename:
                try:
                    export_dir = path.join(getcwd(), "exports")
                    makedirs(export_dir, exist_ok=True)

                    csv_path = path.join(export_dir, filename + ".csv")
                    tsp_path = path.join(export_dir, filename + ".tsp")

                    self.problem.extract_coords_to_csv(csv_path)
                    self.log(f"Success! Coordinates exported to {csv_path}")

                    self.problem.extract_coords_to_tsp(tsp_path)
                    self.log(f"Success! Coordinates exported to {tsp_path}")

                    popup.destroy()
                except Exception as e:
                    self.log(f"Error {e}")
            else:
                self.log("Input Required. Please enter a filename.")

        popup = ctk.CTkToplevel()
        popup.title("Export Coordinates")

        label = ctk.CTkLabel(popup, text="Enter filename for export:")
        label.pack(padx=10, pady=10)

        entry = ctk.CTkEntry(popup, width=200)
        entry.pack(padx=10)

        submit_button = ctk.CTkButton(popup, text="Export", command=submit_filename)
        submit_button.pack(pady=10)

        popup.grab_set()

    def redirect_stdout(self):
        import sys
        import io

        class StdoutRedirector(io.TextIOBase):
            def __init__(inner_self, log_func):
                inner_self.log_func = log_func

            def write(inner_self, s):
                if s.strip():
                    inner_self.log_func(s.strip())

            def flush(inner_self):
                pass

        sys.stdout = StdoutRedirector(self.log)

    def clear_log(self):
        self.log("Called clear_log")
        self.terminal.delete("1.0", "end")
        self.progress_index = None

    def log(self, text):
        self.terminal.insert("end", text + "\n")
        self.terminal.see("end")


if __name__ == "__main__":
    app = TSPSolverApp()
    app.mainloop()

