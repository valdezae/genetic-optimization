import random

from scipy.spatial import distance


def riesz_s_energy(a_prime, s, size):
    result = 0
    for i in range(size):
        for j in range(size):
            if i != j:
                dist = distance.euclidean(a_prime[i], a_prime[j])
                result += 1 / dist ** s
    return result


def fitness_function(points, size):
    if size < 2:
        return 0

    min_distance = float('inf')

    for i in range(size):
        for j in range(i + 1, size):
            p1 = points[i]
            p2 = points[j]
            # Euclidean distance
            dist = distance.euclidean(p1, p2)
            min_distance = min(min_distance, dist)

    return min_distance


def create_initial_population(population_size, path):

    with open(path, 'r') as file:

        parameters = file.readline().strip().split()
        instance_size = int(parameters[1])
        dimensions = int(parameters[2])
        lines = file.readlines()

    points = []
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

    population = random.sample(points, population_size)
    return population


def selection(population, fitnesses, tournament_size):
    print()


def main():
    population = create_initial_population(50, "D:/PyCharm/GeneticOpt/Instancias/ZCAT1_200_02D.pof")


if __name__ == "__main__":
    main()
