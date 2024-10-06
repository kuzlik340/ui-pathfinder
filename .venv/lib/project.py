import math
import random
from random import randint, uniform, shuffle, sample
import matplotlib.pyplot as plt


#Function that were written by ChatGPT
def draw_map(cities, route):
    plt.figure(figsize=(8, 8))
    x_coords = [city[0] for city in cities]
    y_coords = [city[1] for city in cities]
    plt.scatter(x_coords, y_coords, c='blue', marker='o', s=100, label='Cities')
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, f'{i}', fontsize=12, ha='right')
    for i in range(len(route) - 1):
        city_index1 = route[i]
        city_index2 = route[i + 1]
        plt.plot([cities[city_index1][0], cities[city_index2][0]],
                 [cities[city_index1][1], cities[city_index2][1]], c='green', lw=2)
    first_city_index = route[0]
    last_city_index = route[-1]
    plt.plot([cities[last_city_index][0], cities[first_city_index][0]],
             [cities[last_city_index][1], cities[first_city_index][1]], c='green', lw=2, label='Route')
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.grid(True)
    plt.legend()
    plt.title('City Map and Route')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.show()


#function for open and read file into sub-array
def openfile():
    with open('input.txt', 'r') as file:
        data = file.read()

    data = data.replace('(', '').replace(')', '')
    pairs = data.split(', ')

    cities = []
    for i in range(0, len(pairs), 2):
        x = int(pairs[i])
        y = int(pairs[i + 1])
        cities.append([x, y])

    return cities


def mut_swap_two_neighbours(arr):
    ind = randint(0, len(arr) - 2)
    arr[ind], arr[ind + 1] = arr[ind + 1], arr[ind]

#swap two random indexes
def mut_swap_two_rand(arr):
    ind1 = randint(0, len(arr) - 1)
    ind2 = randint(0, len(arr) - 1)
    arr[ind1], arr[ind2] = arr[ind2], arr[ind1]

#reverse sub-array in individual
def mut_reverse_subarray(arr):
    start = randint(0, len(arr) - 2)
    end = randint(start + 1, len(arr) - 1)
    arr[start:end] = arr[start:end][::-1]

#shuffle sub-array in individual
def mut_shuffle_subarray(arr):
    start = randint(0, len(arr) - 2)
    end = randint(start + 1, len(arr) - 1)
    subarray = arr[start:end]
    shuffle(subarray)
    arr[start:end] = subarray

#function for mutate individual
def mutation(arr):
    mutation_type_chance = randint(0, 100)
    if mutation_type_chance > 75:
        mut_swap_two_neighbours(arr)
    elif mutation_type_chance > 50:
        mut_reverse_subarray(arr)
    elif mutation_type_chance > 30:
        mut_shuffle_subarray(arr)
    else:
        mut_swap_two_rand(arr)

#function to compute distance
def dist(x1, y1, x2, y2):
    return math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))


def calculate_fitness(individuals, cities):
    fitness = []
    for individual in individuals:
        fit_tmp = 0
        #compute fitness in individual
        #individuals is an array of all individuals in generation
        #1 individual is a vector of indexes
        for i in range(len(individual) - 1):
            city_index1 = individual[i]
            city_index2 = individual[i + 1]
            fit_tmp += dist(cities[city_index1][0], cities[city_index1][1], cities[city_index2][0], cities[city_index2][1])
        #closing our distance by adding to distance the distance between last two cities
        first_city_index = individual[0]
        last_city_index = individual[-1]
        fit_tmp += dist(cities[last_city_index][0], cities[last_city_index][1], cities[first_city_index][0],cities[first_city_index][1])
        fitness.append(fit_tmp)
    return fitness


def roulette_wheel_selection(population, fitness_scores):
    #invert all fitness cause rank selection works good if you want to find a larger fitness
    inverted_fitness = [1 / fitness if fitness > 0 else float('inf') for fitness in fitness_scores]
    total_fitness = sum(inverted_fitness)
    #Picking the number between zero and total fitness of population
    # to find the border when we will return individual
    pick = uniform(0, total_fitness)
    current = 0
    #Creating 2 dimension array from population and inverted fitness
    # to sum up all the fitness until it will be more than our variable pick
    for individual, fitness in zip(population, inverted_fitness):
        current += fitness
        if current >= pick:
            return individual


import random


def tournament_selection(population, fitness_scores, tournament_size):
    tournament = []
    # Append random individuals to a tournament selection
    for _ in range(tournament_size):
        index = random.randint(0, len(population) - 1)
        tournament.append((population[index], fitness_scores[index]))

    tournament_winner = tournament[0]

    # Checking every individual in tournament array
    for individual, fitness in tournament[1:]:
        # Check fitness of current winner with the fitness in tournament array
        if fitness < tournament_winner[1]:
            tournament_winner = (individual, fitness)

    return tournament_winner[0]


def elitism_selection(population, fitness_scores, elite_size):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
    return sorted_population[:elite_size]


def crossover(parent1, parent2):
    #Initializing empty child
    child = [None] * len(parent1)

    cut1 = randint(0, len(parent1) - 1)
    cut2 = randint(0, len(parent1) - 1)

    if cut1 > cut2:
        cut1, cut2 = cut2, cut1

    #copying segment from one parent
    child[cut1:cut2] = parent1[cut1:cut2]

    #copying other two parts from another parent
    current_pos = cut2
    for city in parent2:
        if city not in child:
            if current_pos == len(child):
                current_pos = 0
            child[current_pos] = city
            current_pos += 1
    return child

def replace_random_individuals(population, num_individuals, indexes):
    for _ in range(num_individuals):
        new_individual = indexes[:]
        shuffle(new_individual)

        replace_index = randint(0, len(population) - 1)

        population[replace_index] = new_individual

def create_new_generation(population, fitness_scores, elite_size, mutation_rate, tournament_size):
    new_generation = []

    #appending elite individuals without any mutations
    elite_individuals = elitism_selection(population, fitness_scores, elite_size)
    new_generation.extend(elite_individuals)

    while len(new_generation) < len(population):
        parent1 = roulette_wheel_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores, tournament_size)
        child = crossover(parent1, parent2)
        if random.random() < mutation_rate:
           mutation(child)
        #an optimisation feature so the generation
        # won't have too many same individuals
        if child not in new_generation:
            new_generation.append(child)
        else:
            child = crossover(parent1, parent2)
            mutation(child)
            new_generation.append(child)

    return new_generation


def start1(population_size, mutation_rate, elite_size, amount_of_generations, tournament_size):
    individuals = []
    cities = openfile()
    indexes = []
    for i in range(len(cities)):
        indexes.append(i)

    #creating first generation
    for _ in range(population_size):
        individual = indexes[:]
        shuffle(individual)
        individuals.append(individual)

    #computing fitness of first generation
    fitness_scores = calculate_fitness(individuals, cities)
    stagnation_counter = 0
    best_fitness = float('inf')
    last_best_fitness = best_fitness
    best_route = None
    result_fitness = float('inf')
    result_individual = None
    for generation in range(amount_of_generations):

        individuals = create_new_generation(individuals, fitness_scores, elite_size, mutation_rate, tournament_size)

        #Computing fitness and finding the best one
        fitness_scores = calculate_fitness(individuals, cities)
        best_fitness = min(fitness_scores)

        best_index = fitness_scores.index(best_fitness)
        best_route = individuals[best_index]

        if best_fitness < result_fitness:
            result_fitness = best_fitness
            result_individual = best_route

        if best_fitness == last_best_fitness:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        if stagnation_counter >= 7:
            replace_random_individuals(individuals, 20, indexes)
        last_best_fitness = best_fitness
    print(f"Best Route: {result_individual} | Best Fitness: {result_fitness}")
    draw_map(cities, best_route)  # Рисуем карту с лучшим маршрутом
    return best_fitness












