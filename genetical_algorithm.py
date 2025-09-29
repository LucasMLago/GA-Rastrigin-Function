import random 
import numpy as np
import pandas as pd
from typing import Callable, Tuple, List, Optional, Literal

from geneticalgorithm2 import AlgorithmParams
from geneticalgorithm2 import GeneticAlgorithm2 as ga

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LinearLocator
from mpl_toolkits.mplot3d import Axes3D

def plot_function(
        f: Callable, 
        dim: Literal[2, 3],
        boundaries: Tuple[float, float, float, float], 
        points: int = 101, 
        colormap: str = "twilight", 
        title: str = "Function Plot",
        mode: Literal["min", "max"] = "min",
        output: str = "output"
    ):

    """
    Plots the given function f in 2D (Contour Plot) or 3D (Surface Plot).

    Parameters:
    -----------
    
    f : Callable
        The function to be plotted. It should take a list or array of length 2 as input (e.g., f([x, y])).
    
    dim : Literal[2, 3]
        The dimension of the visualization. 2 for Contour Plot (2D) or 3 for Surface Plot (3D).
    
    boundaries : Tuple[float, float, float, float]
        The boundaries for the input variables: (xmin, xmax, ymin, ymax).

    points : int
        The number of points to be used in each dimension for plotting.
    
    colormap : str
        The colormap to be used for plotting.
    
    title : str
        The title of the plot.

    mode : Literal["min", "max"]
        Whether to find the minimum or maximum point of the function for annotation.
    
    output : str
        The base filename for saving the plots (without extension).
    """

    xmin, xmax, ymin, ymax = boundaries
    x = np.linspace(xmin, xmax, points)
    y = np.linspace(ymin, ymax, points)
    X, Y = np.meshgrid(x, y)
    
    Z = np.array([[f(np.array([x_val, y_val])) for x_val in x] for y_val in y])

    if mode == "max":
        idx = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
    elif mode == "min":
        idx = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
    else:
        raise ValueError("Mode must be either 'min' or 'max'")
    
    best_y = y[idx[0]]
    best_x = x[idx[1]]
    best_val = Z[idx]

    levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())

    subtitle = f"Best {mode}: f({best_x:.2f}, {best_y:.2f}) = {best_val:.2f}"
    full_title = r"$\bf{" + title + r"}$" + "\n" + subtitle
    
    if dim == 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        cp = ax.contourf(X, Y, Z, levels=levels, cmap=colormap)
        fig.colorbar(cp, ax=ax, label="Function Value")
        
        ax.set_title(full_title, fontsize=14)
        ax.set_xlabel('X-axis', fontsize=12)
        ax.set_ylabel('Y-axis', fontsize=12)
        ax.axis([xmin, xmax, ymin, ymax])
        
        fig.tight_layout()
        plt.savefig(f"{output}.png", dpi=250)
        plt.close(fig)

    elif dim == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap=colormap, linewidth=0, antialiased=False, alpha=0.9)
        fig.colorbar(surf, shrink=0.6, aspect=20, label="Function Value")
        
        ax.contour(
            X, Y, Z, 
            zdir='z', 
            offset=Z.min(), 
            levels=levels, 
            cmap=colormap,
            linewidths=1.2
        )

        ax.set_zlim(Z.min(), Z.max())
        ax.zaxis.set_major_locator(LinearLocator(4))
        ax.view_init(elev=35, azim=-120)
        
        ax.set_title(full_title, fontsize=14, pad=10)
        ax.set_xlabel('X-axis', fontsize=12)
        ax.set_ylabel('Y-axis', fontsize=12)
        ax.set_zlabel('Z-axis', fontsize=12)
        
        fig.tight_layout()
        plt.savefig(f"{output}.png", dpi=250)
        plt.close(fig)
    
    else:
        raise ValueError("Dimension must be either 2 or 3")
    
def ga_plot(experiments: List[np.float64], title: str = "GA Experiments", scale: str = "linear", output: str = "GA Experiments"):
    plt.figure(figsize=(10, 7))
    for i, exp in enumerate(experiments):
        plt.plot(exp, label=f'Experiment {i+1}')
    plt.xlabel('Generation')
    plt.ylabel('Minimized function')
    plt.title(title)

    if scale in ['linear', 'log', 'symlog', 'logit']:
        plt.yscale(scale)
    else:
        raise ValueError("Scale must be one of 'linear', 'log', 'symlog', 'logit'")

    plt.legend()
    plt.savefig(f"{output}.png")

def plot_comparison(
        f: Callable,
        dim: int,
        bounds: Tuple[float, float, float, float],
        ga_experiments: List[np.float64],
        iterations: int = 1000,
        scale: str = "linear",
        output: str = "comparison"
    ):
    """
    Compares the best GA experiment with Random Search and Gradient Descent.
    """

    best_ga = min(ga_experiments, key=lambda exp: exp[-1])

    # Random Search
    rs_history = random_search(f, dim, bounds, iterations)

    # Gradient Descent
    def grad_f(x):
        A = 10
        return 2 * np.array(x) + 2 * A * np.pi * np.sin(2 * np.pi * np.array(x))

    gd_history = gradient_descent(
        f, grad_f, np.random.uniform(bounds[0], bounds[1], dim), iterations=iterations
    )

    # Plot comparison
    plt.figure(figsize=(10, 7))

    plt.plot(best_ga, label="Best GA", color="blue", linewidth=2)
    plt.plot(rs_history, label="Random Search", color="green", linewidth=2)
    plt.plot(gd_history, label="Gradient Descent", color="orange", linewidth=2)

    plt.xlabel("Iteration / Generation")
    plt.ylabel("Best Function Value")
    plt.title("Comparison of Best GA vs Random Search vs Gradient Descent")
    if scale in ['linear', 'log', 'symlog', 'logit']:
        plt.yscale(scale)
    else:
        raise ValueError("Scale must be one of 'linear', 'log', 'symlog', 'logit'")
    plt.legend()
    plt.savefig(f"{output}.png")
    plt.close()

def select_parameters() -> AlgorithmParams:
    """
    Selects a valid random combination of parameters for the GA.
    """
    population_size_list = [100, 200, 300, 400]
    mutation_probability_list = [0.001, 0.05, 0.01]
    elit_ratio_list = [0, 0.1, 0.2, 0.25]
    parents_portion_list = [0.25, 0.3, 0.35, 0.4]
    crossover_type_list = ["one_point", "two_point", "uniform", "segment", "shuffle"]
    mutation_type_list = ["uniform_by_x", "uniform_by_center", "gauss_by_center", "gauss_by_x"]
    selection_type_list = ["fully_random", "roulette", "stochastic", "sigma_scaling", "ranking", "linear_ranking", "tournament"]

    while True:
        population_size = random.choice(population_size_list)
        mutation_probability = random.choice(mutation_probability_list)
        elit_ratio = random.choice(elit_ratio_list)
        parents_portion = random.choice(parents_portion_list)
        crossover_type = random.choice(crossover_type_list)
        mutation_type = random.choice(mutation_type_list)
        selection_type = random.choice(selection_type_list)

        elitism_count = int(population_size * elit_ratio)
        parents_count = int(population_size * parents_portion)

        if elitism_count >= parents_count:
            continue
        if elitism_count < 0 or parents_count < 0:
            continue
        if population_size <= 0:
            continue

        return AlgorithmParams(
            max_num_iteration=None,
            max_iteration_without_improv=100,
            population_size=population_size,
            mutation_probability=mutation_probability,
            elit_ratio=elit_ratio,
            parents_portion=parents_portion,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            selection_type=selection_type,
        )

def genetical_algorithm(
        f: Callable,
        dim: int,
        bounds: Tuple[float, float, float, float],
        number_of_experiments: int = 5,
        initial_point: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, List[np.float64]]:

    """
    Runs multiple experiments of a genetic algorithm on the given function f.

    Parameters:
    -----------

    f : function
        The function to be minimized. It should take a list or array as input and return a scalar value.
    
    dim : int
        The dimension of the input space.

    bounds : Tuple[float, float, float, float]
        The boundaries for the input variables -> (min, max).

    number_of_experiments : int
        The number of experiments to run.

    initial_point : Optional[np.ndarray]
        The starting point for the optimization.

    Returns:
    --------

    df : pd.DataFrame
        A DataFrame containing the parameters and results of each experiment.
    
    experiments : List[np.float64]
        A list of arrays, each containing the convergence history of an experiment.
    """
    df = pd.DataFrame()
    experiments = []
    for i in range(0, number_of_experiments):
        algorithm_parameters = select_parameters()

        model = ga(
            dimension=dim,
            variable_type="real",
            variable_boundaries=[(bounds[0], bounds[1])] * dim,
            algorithm_parameters=algorithm_parameters
        )

        # Set the initial generation if provided
        start_generation = None
        if initial_point is not None:
            start_generation = np.tile(initial_point, (algorithm_parameters.population_size, 1))

        model.run(function=f, no_plot=True, disable_printing=True, start_generation=start_generation)
        experiments.append(model.report)
        convergence_generation = len(model.report)

        exp_dict = {
            "experiment": i+1,
            "population_size": algorithm_parameters.population_size,
            "mutation_probability": algorithm_parameters.mutation_probability,
            "elit_ratio": algorithm_parameters.elit_ratio,
            "parents_portion": algorithm_parameters.parents_portion,
            "crossover_type": algorithm_parameters.crossover_type,
            "mutation_type": algorithm_parameters.mutation_type,
            "selection_type": algorithm_parameters.selection_type,
            "convergence_generation": convergence_generation,
            "best_function_value": model.best_function,
            "mean": np.mean(model.report),
            "std": np.std(model.report),
            "min": np.min(model.report),
            "max": np.max(model.report)
        }

        df = pd.concat([df, pd.DataFrame([exp_dict])], ignore_index=True)

    return df, experiments

def random_search(
        f: Callable,
        dim: int,
        bounds: Tuple[float, float, float, float],
        iterations: int = 1000,
        initial_point: Optional[np.ndarray] = None
    ):
    best_val = float("inf")
    history = []
    x = initial_point if initial_point is not None else np.random.uniform(bounds[0], bounds[1], dim)
    for _ in range(iterations):
        val = f(x)
        if val < best_val:
            best_val = val
        history.append(best_val)
        x = np.random.uniform(bounds[0], bounds[1], dim)
    return history

def gradient_descent(
        f: Callable, 
        grad_f: Callable, 
        x0: np.ndarray, 
        learning_rate: float = 0.01, 
        iterations: int = 1000
    ):
    x = x0
    history = []
    for _ in range(iterations):
        grad = grad_f(x)
        x = x - learning_rate * grad
        history.append(f(x))
    return history

def run_pipeline():
    """
    Executes the full pipeline: runs the genetic algorithm, compares it with other methods, and plots results.
    """

    def rastrigin(x):
        """
        Rastrigin function is a non-convex 
        function used as a performance test 
        problem for optimization algorithms.

        It is a typical example of a non-linear
        multimodal function.
        """

        A = 10
        dim = len(x)
        return A * dim + sum([(xi**2 - A * np.cos(2 *np.pi * xi)) for xi in x])

    EXPERIMENTS = 10
    BOUNDS = (-5.12, 5.12, -5.12, 5.12)
    INITIAL_POINT = np.random.uniform(BOUNDS[0], BOUNDS[1], 3)

    plot_function(rastrigin, dim=2, boundaries=BOUNDS, title="Rastrigin Function (2D)", output="output/function_plot/rastrigin_2D")
    plot_function(rastrigin, dim=3, boundaries=BOUNDS, title="Rastrigin Function (3D)", output="output/function_plot/rastrigin_3D")

    df, experiments = genetical_algorithm(f=rastrigin, bounds=BOUNDS, dim=3, number_of_experiments=EXPERIMENTS, initial_point=INITIAL_POINT)

    ga_plot(experiments, title="GA Experiments (linear)", scale="linear", output="output/ga_plot/ga_experiments_linear")
    ga_plot(experiments, title="GA Experiments (log)", scale="log", output="output/ga_plot/ga_experiments_log")

    plot_comparison(f=rastrigin, bounds=BOUNDS, dim=3, ga_experiments=experiments, scale="linear", output="output/comparison_plot/rastrigin_linear")
    plot_comparison(f=rastrigin, bounds=BOUNDS, dim=3, ga_experiments=experiments, scale="log", output="output/comparison_plot/rastrigin_log")

    df.to_csv("output/ga_results.csv", index=False)

if __name__ == "__main__":
    run_pipeline()