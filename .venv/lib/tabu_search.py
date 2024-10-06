import math
import random
import matplotlib.pyplot as plt
from random import randint, shuffle


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

# Same fitness function that is used for genetic algorithm
def calculate_fitness(individual, cities):
    fit_tmp = 0
    for i in range(len(individual) - 1):
        city_index1 = individual[i]
        city_index2 = individual[i + 1]
        fit_tmp += dist(cities[city_index1][0], cities[city_index1][1], cities[city_index2][0], cities[city_index2][1])
    first_city_index = individual[0]
    last_city_index = individual[-1]
    fit_tmp += dist(cities[last_city_index][0], cities[last_city_index][1], cities[first_city_index][0],
                    cities[first_city_index][1])
    return fit_tmp


def dist(x1, y1, x2, y2):
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


#All of the functions for making neighbors are stolen from the genetic algorithm
#in genetic algorithm they were used for mutation
def swap_two_neighbours(arr):
    ind = randint(0, len(arr) - 2)
    arr[ind], arr[ind + 1] = arr[ind + 1], arr[ind]

#swap two random indexes
def swap_two_rand(arr):
    ind1 = randint(0, len(arr) - 1)
    ind2 = randint(0, len(arr) - 1)
    arr[ind1], arr[ind2] = arr[ind2], arr[ind1]

#reverse sub-array in individual
def reverse_subarray(arr):
    start = randint(0, len(arr) - 2)
    end = randint(start + 1, len(arr) - 1)
    arr[start:end] = arr[start:end][::-1]

#shuffle sub-array in individual
def shuffle_subarray(arr):
    start = randint(0, len(arr) - 2)
    end = randint(start + 1, len(arr) - 1)
    subarray = arr[start:end]
    shuffle(subarray)
    arr[start:end] = subarray

#function for make neighbor that will randomly make some changes
# to the previous solution
def make_neighbor(arr):
    type_of_creating_neighbor = randint(0, 100)
    if type_of_creating_neighbor > 75:
        swap_two_neighbours(arr)
    elif type_of_creating_neighbor > 50:
        reverse_subarray(arr)
    elif type_of_creating_neighbor > 30:
        shuffle_subarray(arr)
    else:
        swap_two_rand(arr)
    return arr


def tabu_search(max_iterations, tabu_size):
    cities = openfile()
    solution = []
    for i in range(len(cities)):
      solution.append(i)
    shuffle(solution)
    fitness = calculate_fitness(solution, cities)
    best_fitness = fitness
    best_solution = solution
    tabu_list = [solution]
    for iteration in range(max_iterations):
        neighbors = []
        neighbor_fitness = []
        #Generating neighbors for the solution
        for _ in range(len(cities) * 5):
            new_solution = make_neighbor(solution[:])
            if new_solution not in tabu_list:  # check if not in tabu-list
                neighbors.append(new_solution)
                neighbor_fitness.append(calculate_fitness(new_solution, cities))

        best_index = neighbor_fitness.index(min(neighbor_fitness))
        best_neighbour_solution = neighbors[best_index]
        best_neighbour_fitness = neighbor_fitness[best_index]

        #append all solutions cause we do not want to deal with them again because
        # they will make similar results so the algorithm will stagnate
        tabu_list.append(best_neighbour_solution)
        # delete oldest element if tabu-list is full
        if len(tabu_list) >= tabu_size:
                tabu_list.pop(0)
        solution = best_neighbour_solution
        fitness = best_neighbour_fitness

        # always check for the best fitness and best solution
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = solution

    draw_map(cities, best_solution)
    return best_solution, best_fitness




