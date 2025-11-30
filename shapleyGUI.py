import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import subplots
from matplotlib import style
from threading import Thread
from time import time
from TSPsolver.tsp_problem_gen import TSPProblem
from TSPsolver.tspga_config import GAConfig
from Shapley.shapley import solve_Shapley, solve_Shapley_prl
from tkinter import filedialog
from pandas import read_csv


class ShapleySolverApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Shapley values solver")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("green")

        style.use("dark_background")  

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
        self.num_samples = 100

        self.redirect_stdout()

    def build_config_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(side="left", fill="y", padx=10, pady=10)

        self.entries = {}
        config_params = [
            ("Cities", 20), ("Number of Samples", 100), ("Generations", 1000), ("No Improvement Limit", 20),
            ("Mutation Rate", 0.1), ("Elitism Rate", 0.1), ("Selection Ratio", 0.5), ("Size Parameter", 10)
        ]

        self.param_bounds = {
            "Cities": (3, 20),
            "Number of Samples": (10, 2000),
            "Generations": (1, 1000),
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

        ctk.CTkButton(frame, text="Random Problem", command=self.generate_problem).pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Load Problem", command=self.load_problem_from_csv).pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Start Serial", command=lambda: self.run_solver(mode='serial')).pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Start Parallel", command=lambda: self.run_solver(mode='parallel')).pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Clear Log", command=self.clear_log).pack(side="left", padx=5)

        return frame

    def build_plot_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        plot_frame = ctk.CTkFrame(frame, corner_radius=20, fg_color="#1e1e1e")
        plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.fig, self.ax = subplots(figsize=(5, 4))
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_title("Best Tour")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.get_tk_widget().configure(bg='#1e1e1e', highlightthickness=0, bd=0)

        return frame

    def plot_cities(self, tour: tuple[int] = ()):
        self.ax.clear()
        self.ax.set_title("Best Tour")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        coords = self.problem.coordinates
        x_all = [c[0] for c in coords]
        y_all = [c[1] for c in coords]

        self.ax.scatter(x_all[1:], y_all[1:], c='#00FFA3', edgecolors='green')
        self.ax.scatter(x_all[0], y_all[0], c='#FF69B4', zorder=5)

        for i, (cx, cy) in enumerate(coords):
            self.ax.text(cx, cy, str(i), fontsize=8, ha='right', va='bottom')

        if tour and len(tour) >= 2:
            for i in range(len(tour) - 1):
                c1 = coords[tour[i]]
                c2 = coords[tour[i + 1]]
                self.ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color='#00BFFF', linewidth=1.5)

        self.fig.tight_layout()
        self.canvas.draw()

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

        self.terminal = ctk.CTkTextbox(frame, height=300)
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

        return val 

    def save_configuration(self):
        clipped_values = {}

        for label, entry in self.entries.items():
            clipped_value = self.clip_and_correct_entry(label, entry)
            clipped_values[label] = clipped_value
        
        self.num_samples = clipped_values["Number of Samples"]

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
        start = time()
        if mode == 'serial':
            shapley_values, tspSol, tour = solve_Shapley(self.problem, self.config, self.num_samples)
        else:
            self.log("Computing Shapley vlues in parallel, wait...")
            shapley_values, tspSol, tour = solve_Shapley_prl(self.problem, self.config, self.num_samples)
        end = time()
        
        sumShap = 0
        for i in shapley_values:
            sumShap += shapley_values[i]
        self.log(f"\n=== Solution ===")
        self.log(f"Shapley values for given problem are: {shapley_values}")
        self.log(f"TSP length for grand tour: {tspSol:.3f} sum of shapley values {sumShap:.3f}")
        self.log(f"Grand tour: {tour}")
        self.log(f"Elapsed time: {end - start:.3f} s")
        self.log(f"time per sample: {(end - start)/self.num_samples:.5f}s")

        self.plot_cities(tour + (0,))

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
            if len(df['x']) > 20:
                self.log("Problem is too big! Suported size is (3-20)!")
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

    def redirect_stdout(self):
        import sys
        from types import SimpleNamespace

        def write(s):
            if '\r' in s:
                self.log(s.strip().lstrip('\r'), overwrite_last_line=True)
            elif s.strip():
                self.log(s.strip())

        def flush():
            pass

        sys.stdout = SimpleNamespace(write=write, flush=flush)

    def clear_log(self):
        self.log("Called clear_log")
        self.terminal.delete("1.0", "end")
        self.progress_index = None

    def log(self, text, overwrite_last_line=False):
        if overwrite_last_line:
            if hasattr(self, 'progress_index'):
                self.terminal.delete(self.progress_index, f'{self.progress_index} lineend')
            else:
                self.progress_index = self.terminal.index("end-1c")
            self.terminal.insert(self.progress_index, text)
        else:
            self.terminal.insert("end", text + "\n")
            self.progress_index = self.terminal.index("end-1c")
        self.terminal.see("end")

if __name__ == "__main__":
    app = ShapleySolverApp()
    app.mainloop()

