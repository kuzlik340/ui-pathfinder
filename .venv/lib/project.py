import math
from random import random, randint, shuffle


def openfile():
    with open('input.txt', 'r') as file:
        data = file.read()

    data = data.replace('(', '').replace(')', '')
    pairs = data.split(', ')

    cities = []
    index = 0  # Индекс города
    for i in range(0, len(pairs), 2):
        x = int(pairs[i])
        y = int(pairs[i + 1])
        # Добавляем к координатам индекс города
        cities.append([index, x, y])
        index += 1  # Увеличиваем индекс для следующего города
    return cities




def first_shuffle(arr):
    shuffle(arr)


def mutation(arr):
    chos = randint(0, 100)
    if chos <= 95:
        ind = randint(0, len(arr)-2)
        arr[ind], arr[ind+1] = arr[ind+1], arr[ind]
    else:
        ind1 = randint(0, len(arr) - 1)
        ind2 = randint(0, len(arr) - 1)
        arr[ind1], arr[ind2] = arr[ind2], arr[ind1]

def dist(x1, y1, x2, y2):
    return sqrt(((x1 - x2)**2) + ((y1 - y2)**2))

def calculate_fitness(routes, cities):
    fitness = []
    for route in routes:
        fit_tmp = 0
        for i in range(len(cities)-2):
            fit_tmp += dist(cities[i][0], cities[i][1], cities[i + 1][0], cities[i + 1][1])
        fitness.append(fit_tmp)
    return fitness


def main():
    all_shuffled_routes = []
    cities = openfile()
    indices = [city[0] for city in cities]

    for _ in range(20):
        shuffled_indices = indices[:]
        first_shuffle(shuffled_indices)  # Перемешиваем копию
        all_shuffled_routes.append(shuffled_indices)  # Добавляем в общий список

    # Печатаем все 20 массивов
    for i, route in enumerate(all_shuffled_routes):
        print(f"Route {i + 1}: {route}")

if __name__ == "__main__":
    main()






