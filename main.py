import os
import random
import time
import itertools
import matplotlib.pyplot as plt

from scipy.spatial import distance
from prettytable import PrettyTable
from math import comb


def riesz_s_energy(a_prime, s, collision_penalty=1e6):
    result = 0
    for i in range(len(a_prime)):
        for j in range(len(a_prime)):
            if i != j:
                dist = distance.euclidean(a_prime[i], a_prime[j])
                if dist == 0:
                    result += collision_penalty  # Large penalty for overlap
                else:
                    result += 1 / (dist ** s)  # Avoid division by 0
    return result


def create_initial_population(population_size, individual_size, path):
    points = []

    with open(path, 'r') as file:
        parameters = file.readline().strip().split()
        instance_size = int(parameters[1])
        dimensions = int(parameters[2])
        lines = file.readlines()

    if dimensions == 2:
        for line in lines:
            values = line.strip().split()
            x = float(values[0])
            y = float(values[1])
            points.append((x, y))
    else:
        for line in lines:
            values = line.strip().split()
            x = float(values[0])
            y = float(values[1])
            z = float(values[2])
            points.append((x, y, z))

    population = []

    for _ in range(population_size):
        random.shuffle(points)  # Shuffle to increase variation
        individual = random.sample(points, individual_size)  # Take first individual_size elements
        population.append(individual.copy())  # Store a copy to prevent modifications

    return population, points, dimensions


def selection(population, fitnesses, tournament_size):
    selected = []
    for _ in range(len(population)):
        # Crear índices para el torneo
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        # Seleccionar al individuo con mejor fitness del torneo
        winner_idx = min(tournament_indices, key=lambda idx: fitnesses[idx])
        selected.append(population[winner_idx].copy())  # Importante hacer una copia
    return selected


def crossover(parent1, parent2):
    # Creamos listas vacías para los hijos
    child1 = []
    child2 = []

    # Para cada posición en los padres
    for i in range(len(parent1)):
        # Decidimos aleatoriamente qué padre contribuye a qué hijo
        if random.random() < 0.5:
            # El punto del primer padre va al primer hijo y el del segundo al segundo hijo
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            # El punto del segundo padre va al primer hijo y el del primero al segundo hijo
            child1.append(parent2[i])
            child2.append(parent1[i])

    return child1, child2


def mutation(individual, mutation_rate, all_points):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            # Reemplazamos el punto con uno aleatorio del conjunto de todos los puntos
            mutated_individual[i] = random.choice(all_points)
    return mutated_individual


def plot_individual(ax, individual, title, dimensions):
    ax.clear()
    if dimensions == 2:
        x = [point[0] for point in individual]
        y = [point[1] for point in individual]
        ax.scatter(x, y, color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:  # 3D
        ax.remove()
        ax = plt.gcf().add_subplot(111, projection='3d')
        x = [point[0] for point in individual]
        y = [point[1] for point in individual]
        z = [point[2] for point in individual]
        ax.scatter(x, y, z, color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=25, azim=45)  # Rotate to front-facing

    ax.set_title(title)
    return ax


def genetic_algorithm(population_size, individual_size, generations, mutation_rate, path, output_dir):
    population, points, dimensions = create_initial_population(population_size, individual_size, path)
    #print(population)

    if dimensions == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:  # 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    best_performers = []

    all_populations = []

    table = PrettyTable()
    table.field_names = ["Generation", "Fitness"]

    overall_best_individual = None
    overall_best_fitness = float('inf')
    previous_generation_best_fitness = None
    fitness_repetion = 0
    for generation in range(generations):
        fitnesses = [riesz_s_energy(ind, dimensions + 1) for ind in population]


        best_idx = fitnesses.index(min(fitnesses))
        best_individual = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]


        if best_fitness < overall_best_fitness:
            overall_best_fitness = best_fitness
            overall_best_individual = best_individual.copy()

        best_performers.append((best_individual, best_fitness))
        all_populations.append(population[:])
        table.add_row([generation + 1, best_fitness])

        if previous_generation_best_fitness is not None and previous_generation_best_fitness == best_fitness:
            fitness_repetion += 1
            if fitness_repetion == 2:
                break
        else:
            # Reset counter if fitness changed
            fitness_repetion = 0

        previous_generation_best_fitness = best_fitness

        population = selection(population, fitnesses, 10)
        #print(population)

        next_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1]

            child1, child2 = crossover(parent1, parent2)

            next_population.append(mutation(child1, mutation_rate, points))
            next_population.append(mutation(child2, mutation_rate, points))

        next_population[0] = best_individual
        population = next_population

    print(table)

    # Plot the best overall individual
    title = f"Best Individual (Fitness: {overall_best_fitness:.4f})"
    plot_individual(ax, overall_best_individual, title, dimensions)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    file_base = os.path.splitext(os.path.basename(path))[0]
    img_path = os.path.join(output_dir, f"{file_base}_I{individual_size}.png")
    plt.savefig(img_path)
    plt.close()

    return overall_best_fitness


def brute_force(path, individual_size):

    # Read points from the file
    points = []
    with open(path, 'r') as file:
        parameters = file.readline().strip().split()
        instance_size = int(parameters[1])
        dimensions = int(parameters[2])
        lines = file.readlines()

    if dimensions == 2:
        for line in lines:
            values = line.strip().split()
            x = float(values[0])
            y = float(values[1])
            points.append((x, y))
    else:
        for line in lines:
            values = line.strip().split()
            x = float(values[0])
            y = float(values[1])
            z = float(values[2])
            points.append((x, y, z))


    # Calculate total number of combinations

    total_combinations = comb(len(points), individual_size)


    # Generate all possible combinations

    best_individual = None
    best_energy = float('inf')
    count = 0

    # Iterate through ALL possible combinations
    for combination in itertools.combinations(points, individual_size):
        energy = riesz_s_energy(combination, dimensions + 1)
        if energy < best_energy:
            best_energy = energy
            best_individual = combination
            print(f"New best energy found: {best_energy:.4f}")

        count += 1

    print(f"Brute force completed")

    # Create visualization of the best solution
    output_dir = "Results/BruteForce"
    os.makedirs(output_dir, exist_ok=True)

    if dimensions == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:  # 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    title = f"Brute Force Best Solution (Fitness: {best_energy:.4f})"


    if dimensions == 2:
        x = [point[0] for point in best_individual]
        y = [point[1] for point in best_individual]
        ax.scatter(x, y, color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:  # 3D
        x = [point[0] for point in best_individual]
        y = [point[1] for point in best_individual]
        z = [point[2] for point in best_individual]
        ax.scatter(x, y, z, color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=25, azim=45)  # Rotate to front-facing

    ax.set_title(title)

    plt.tight_layout()
    file_base = os.path.splitext(os.path.basename(path))[0]
    img_path = os.path.join(output_dir, f"{file_base}_BF_I{individual_size}.png")
    plt.savefig(img_path)
    plt.close()

    return best_individual, best_energy



def main():
    population_size = 500
    generations = 20
    mutation_rate = 0.1

    input_dir = "Instancias"
    output_dir = "Results/Images"
    log_file = "Results/times.txt"
    brute_force_log = "Results/brute_force_times.txt"

    os.makedirs("Results", exist_ok=True)
    os.makedirs("Results/BruteForce", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    instance_files = [f for f in os.listdir(input_dir) if f.endswith(".pof")]
    individual_sizes = [50,100]

    with open(log_file, 'w') as log:
        log.write("Archivo\tIndividualSize\tTiempo (s)\tFitness\n")
        for individual_size in individual_sizes:
            for file in instance_files:
                path = os.path.join(input_dir, file)
                print(f"Running on: {file} with individual size {individual_size}")
                start_time = time.time()
                fitness = genetic_algorithm(population_size, individual_size, generations, mutation_rate, path, output_dir)
                elapsed = time.time() - start_time
                log.write(f"{file}\t{individual_size}\t{elapsed:.2f}\t{fitness:.4f}\n")
                print(f"Finished {file} [I{individual_size}] in {elapsed:.2f} seconds with fitness {fitness:.4f}")

    # Run brute force
    # with open(brute_force_log, 'w') as log:
    #     log.write("Archivo\tIndividualSize\tTiempo (s)\tFitness\n")
    #     for individual_size in individual_sizes:
    #         #for file in instance_files:
    #         file = "ZCAT1_20_03D.pof"
    #         path = os.path.join(input_dir, file)
    #         print(f"Running brute force on: {file} with individual size {individual_size}")
    #         start_time = time.time()
    #
    #         best_individual, best_energy = brute_force("Instancias/ZCAT1_20_03D.pof", 10)
    #
    #         elapsed = time.time() - start_time
    #         log.write(f"{file}\t{individual_size}\t{elapsed:.2f}\t{best_energy:.4f}\n")
    #         log.flush()  # Ensure data is written immediately
    #         print(f"Finished brute force {file} [I{individual_size}] in {elapsed:.2f} seconds with fitness {best_energy:.4f}")

if __name__ == "__main__":
    main()