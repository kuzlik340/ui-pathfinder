import math
import random
from random import randint, uniform, shuffle, sample


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


def first_shuffle(arr):
    shuffle(arr)


def mutation(arr): #may change parameter for major mutation
    chos = randint(0, 100)
    if chos <= 87 and chos >=80:
        ind = randint(0, len(arr)-2)
        arr[ind], arr[ind+1] = arr[ind+1], arr[ind]
    elif chos < 80 and chos > 50:
        # Выбираем два случайных индекса
        start = randint(0, len(arr) - 2)
        end = randint(start + 1, len(arr) - 1)
        # Инвертируем подмассив между этими индексами
        arr[start:end] = arr[start:end][::-1]
    elif chos <= 40 and chos >30:
        # Выбираем два случайных индекса
        start = randint(0, len(arr) - 2)
        end = randint(start + 1, len(arr) - 1)
        # Перемешиваем подмассив между этими индексами
        subarray = arr[start:end]
        shuffle(subarray)
        arr[start:end] = subarray
    elif chos <=30  and chos >=13:
        # Выбираем случайный элемент для удаления и вставки
        index = randint(0, len(arr) - 1)
        value = arr.pop(index)
        # Выбираем случайный индекс для вставки
        new_index = randint(0, len(arr) - 1)
        arr.insert(new_index, value)
    else:
        ind1 = randint(0, len(arr) - 1)
        ind2 = randint(0, len(arr) - 1)
        arr[ind1], arr[ind2] = arr[ind2], arr[ind1]

def dist(x1, y1, x2, y2):
    return math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))


def calculate_fitness(routes, cities):
    fitness = []

    for route in routes:
        fit_tmp = 0

        for i in range(len(route) - 1):
            city_index1 = route[i]
            city_index2 = route[i + 1]
            fit_tmp += dist(cities[city_index1][0], cities[city_index1][1], cities[city_index2][0], cities[city_index2][1])
        #closing our distance by adding to distance the distance between last two cities
        first_city_index = route[0]
        last_city_index = route[-1]
        fit_tmp += dist(cities[last_city_index][0], cities[last_city_index][1], cities[first_city_index][0],cities[first_city_index][1])

        fitness.append(fit_tmp)
    return fitness


def rank_selection(population, fitness_scores):
    sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1])
    ranks = list(range(1, len(sorted_population) + 1))
    total_rank = sum(ranks)

    pick = uniform(0, total_rank)
    current = 0

    for i, (individual, _) in enumerate(sorted_population):
        current += ranks[i]
        if current > pick:
            return individual

def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    pick = uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitness_scores):
        current += fitness
        if current > pick:
            return individual

def tournament_selection(population, fitness_scores, tournament_size):
    tournament = sample(list(zip(population, fitness_scores)), tournament_size)
    tournament_winner = min(tournament, key=lambda x: x[1])
    return tournament_winner[0]

def elitism_selection(population, fitness_scores, elite_size):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
    return sorted_population[:elite_size]


def crossover(parent1, parent2):
    #Initializing empty child
    child = [None] * len(parent1)

    #
    cut1 = randint(0, len(parent1) - 1)
    cut2 = randint(0, len(parent1) - 1)

    # Обеспечиваем, чтобы cut1 был меньше cut2
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1

    # Шаг 2: Копируем сегмент от первого родителя в ребенка
    child[cut1:cut2] = parent1[cut1:cut2]

    # Шаг 3: Заполняем оставшиеся элементы из второго родителя
    current_pos = cut2
    for city in parent2:
        if city not in child:
            if current_pos >= len(child):
                current_pos = 0
            child[current_pos] = city
            current_pos += 1
    return child


def replace_random_individuals(population, num_individuals, indices):
    for _ in range(num_individuals):
        new_individual = indices[:]
        shuffle(new_individual)

        replace_index = randint(0, len(population) - 1)


        # Заменяем индивидуума в популяции на нового
        population[replace_index] = new_individual


def create_new_generation(population, fitness_scores, elite_size, mutation_rate, tournament_size, mutate_all):
    new_generation = []

    if mutate_all:
        # Мутируем всех лучших индивидов
        best_individuals = elitism_selection(population, fitness_scores, len(population))
        mutate_population(best_individuals)

    # Шаг 1: Добавляем лучших индивидумов (элитизм)
    elite_individuals = elitism_selection(population, fitness_scores, elite_size)
    new_generation.extend(elite_individuals)

    # Шаг 2: Создание потомков через кроссовер
    while len(new_generation) < len(population):
        parent1 = roulette_wheel_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores, tournament_size)
        child = crossover(parent1, parent2)
        if random.random() < mutation_rate:
           mutation(child)
        if child not in new_generation:
            new_generation.append(child)
        else:
            child = crossover(parent1, parent2)
            mutation(child)
            new_generation.append(child)


    return new_generation


def mutate_population(population):
    # Мутируем всех индивидов в популяции
    for individual in population:
        mutation(individual)


def start1(population_size, mutation_rate, elite_size, amount_of_generations, tournament_size):
    all_shuffled_routes = []
    cities = openfile()
    indices = []
    for i in range(len(cities)):
        indices.append(i)

    # Создаем начальную популяцию
    for _ in range(population_size):
        shuffled_indices = indices[:]
        first_shuffle(shuffled_indices)  # Перемешиваем копию
        all_shuffled_routes.append(shuffled_indices)  # Добавляем в общий список

    # Оценка начальной популяции
    fitness_scores = calculate_fitness(all_shuffled_routes, cities)
    before_best_fitness = None
    score_of_gen = 0
    best_fitness = 0  # Лучший фитнес на текущий момент
    stagn_counter = 0

    for generation in range(amount_of_generations):
        # print(f"\nGeneration {generation + 1}")


        all_shuffled_routes = create_new_generation(
                all_shuffled_routes, fitness_scores, elite_size, mutation_rate, tournament_size, False
        )

        # Оценка нового поколения
        fitness_scores = calculate_fitness(all_shuffled_routes, cities)
        best_fitness = min(fitness_scores)

        best_index = fitness_scores.index(best_fitness)
        best_route = all_shuffled_routes[best_index]

        # Проверка на улучшение
        if best_fitness == before_best_fitness:
            stagn_counter += 1
        before_best_fitness = best_fitness
        if stagn_counter >= 3:
            shuffle(all_shuffled_routes[best_index])
            stagn_counter= 0
        # Проверяем, достиг ли лучший фитнес желаемого диапазона
        if best_fitness < 896:  # pu-pu-pu
            score_of_gen += 1
            break
            # Можно прервать цикл, если найден идеальный фитнес
            # break
        # for i, route in enumerate(all_shuffled_routes):
        # print(f"Route {i + 1}: {route}")

        # Выводим лучший маршрут и фитнес
        # print(f"Best Route: {best_route} | Best Fitness: {best_fitness}")

    print(f"Best Fitness: {best_fitness}")
    print(f"Score = {score_of_gen}")
    if score_of_gen == 0:
        return 0
    else:
        return 1












