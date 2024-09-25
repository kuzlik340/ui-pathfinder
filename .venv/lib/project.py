import math
import random
from random import randint, uniform, shuffle, sample


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
    return math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))


def calculate_fitness(routes, cities):
    fitness = []

    for route in routes:
        fit_tmp = 0

        for i in range(len(route) - 1):
            city_index1 = route[i]
            city_index2 = route[i + 1]


            x1, y1 = cities[city_index1][1], cities[city_index1][2]
            x2, y2 = cities[city_index2][1], cities[city_index2][2]


            fit_tmp += dist(x1, y1, x2, y2)


        first_city_index = route[0]
        last_city_index = route[-1]
        x1, y1 = cities[first_city_index][1], cities[first_city_index][2]
        x2, y2 = cities[last_city_index][1], cities[last_city_index][2]
        fit_tmp += dist(x1, y1, x2, y2)

        fitness.append(fit_tmp)

    return fitness


def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    pick = uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitness_scores):
        current += fitness
        if current > pick:
            return individual

# Метод 2: Турнирный отбор
def tournament_selection(population, fitness_scores, tournament_size):
    tournament = sample(list(zip(population, fitness_scores)), tournament_size)
    tournament_winner = min(tournament, key=lambda x: x[1])  # Выбираем лучшего по минимальному фитнесу (длина маршрута)
    return tournament_winner[0]

def elitism_selection(population, fitness_scores, elite_size):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
    return sorted_population[:elite_size]



def crossover(parent1, parent2, cut1, cut2):
    # Создаем пустого потомка с None
    child = [None] * len(parent1)
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1

    # Копируем сегмент из первого родителя
    child[cut1:cut2] = parent1[cut1:cut2]

    # Заполняем оставшиеся города из второго родителя
    pos = cut2
    for city in parent2:
        if city not in child:
            if pos >= len(parent1):  # Циклический сдвиг
                pos = 0
            child[pos] = city
            pos += 1

    return child


def create_new_generation(population, fitness_scores, elite_size, mutation_rate, tournament_size):
    new_generation = []

    # Шаг 1: Добавляем лучших индивидумов (элитизм)
    elite_individuals = elitism_selection(population, fitness_scores, elite_size)
    new_generation.extend(elite_individuals)

    # Шаг 2: Создание потомков через кроссовер
    while len(new_generation) < len(population):
        # Выбираем родителей
        parent1 = roulette_wheel_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores, tournament_size)

        # Генерируем случайные точки разреза
        cut1 = randint(0, len(parent1) - 1)
        cut2 = randint(0, len(parent1) - 1)

        # Скрещивание для создания нового потомка
        child = crossover(parent1, parent2, cut1, cut2)

        # Применение мутации с вероятностью mutation_rate
        if random.random() < mutation_rate:
            mutation(child)

        # Добавляем нового потомка в новое поколение
        new_generation.append(child)

    return new_generation


def start(population_size, mutation_rate, elite_size, amount_of_generations, tournament_size):
    all_shuffled_routes = []
    cities = openfile()
    indices = [city[0] for city in cities]

    # Создаем начальную популяцию
    for _ in range(population_size):
        shuffled_indices = indices[:]
        first_shuffle(shuffled_indices)  # Перемешиваем копию
        all_shuffled_routes.append(shuffled_indices)  # Добавляем в общий список

    # Оценка начальной популяции
    fitness_scores = calculate_fitness(all_shuffled_routes, cities)

    score_of_gen = 0

    best_route_fin = []
    best_fit_fin = 0
    for generation in range(amount_of_generations):
        #print(f"\nGeneration {generation + 1}")

        # Создание новой генерации
        all_shuffled_routes = create_new_generation(all_shuffled_routes, fitness_scores, elite_size,
                                                    mutation_rate, tournament_size)

        # Оценка новой генерации
        fitness_scores = calculate_fitness(all_shuffled_routes, cities)

        # Печатаем лучших индивидумов текущей генерации
        best_fitness = min(fitness_scores)
        best_index = fitness_scores.index(best_fitness)
        if best_fitness >= 895 and best_fitness < 896:
            score_of_gen += 1
        best_route_fin = all_shuffled_routes[best_index]
        best_fit_fin = best_fitness

    print(f"Best Route: {best_route_fin} | Best Fitness: {best_fit_fin}")
    print(f"score  = {score_of_gen}")









