import random

from scipy.spatial import distance
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def riesz_s_energy(a_prime, s):
    result = 0
    for i in range(len(a_prime)):
        for j in range(len(a_prime)):
            if i != j:
                dist = distance.euclidean(a_prime[i], a_prime[j])
                if dist > 0:
                    result += 1 / dist ** s
    return result


def fitness_function(population_points):
    if len(population_points) < 2:
        return 0

    min_distance = float('inf')

    for i in range(len(population_points)):
        for j in range(i + 1, len(population_points)):
            p1 = population_points[i]
            p2 = population_points[j]
            # Euclidean distance
            dist = distance.euclidean(p1, p2)
            min_distance = min(min_distance, dist)

    return min_distance


def create_initial_population(population_size, path):
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
        individual = points[:instance_size]  # Take first `instance_size` elements
        population.append(individual.copy())  # Store a copy to prevent modifications

    return population, points


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

    ax.set_title(title)
    return ax

def genetic_algorithm(population_size, generations, mutation_rate):
    population, points = create_initial_population(population_size, "Instancias/ZCAT1_200_02D.pof")
    #print(population)

    fig, ax = plt.subplots(figsize=(10, 8))

    best_performers = []

    all_populations = []


    table = PrettyTable()
    table.field_names = ["Generation", "Fitness"]

    overall_best_individual = None
    overall_best_fitness = float('inf')

    for generation in range(generations):
        fitnesses = [riesz_s_energy(ind, 3) for ind in population]
        print(fitnesses)

        best_idx = fitnesses.index(min(fitnesses))
        best_individual = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        print(best_fitness)

        if best_fitness < overall_best_fitness:
            overall_best_fitness = best_fitness
            overall_best_individual = best_individual.copy()

        best_performers.append((best_individual, best_fitness))
        all_populations.append(population[:])
        table.add_row([generation + 1, best_fitness])

        population = selection(population, fitnesses, 10)
        #print(population)

        next_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1]

            child1, child2 = crossover(parent1, parent2)

            next_population.append(mutation(child1, mutation_rate, points))
            next_population.append(mutation(child2, mutation_rate, points))

        if random.random() < 0.5:  # 50% chance of keeping or mutating
            next_population[0] = mutation(best_individual, mutation_rate, points)
        else:
            next_population[0] = best_individual
        population = next_population

    print(table)

    # Plot the best overall individual
    title = f"Best Individual (Fitness: {overall_best_fitness:.4f})"
    plot_individual(ax, overall_best_individual, title, 2)

    plt.tight_layout()
    plt.savefig('best_individual.png')
    plt.show()

def main():
    population_size = 50
    generations = 20
    mutation_rate = 0.2

    genetic_algorithm(population_size, generations, mutation_rate)

if __name__ == "__main__":
    main()
