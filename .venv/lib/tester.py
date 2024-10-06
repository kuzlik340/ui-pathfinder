from project import start1
from tabu_search import tabu_search
import random


def main():
    meow = 0
    for i in range(100):
        best_solution, best_fitness = tabu_search(800, 50) #1000,100
        if best_fitness < 896:
            meow += 1
        print(f"Best solution: {best_solution}  |  Fitness: {best_fitness:.2f}")
    print(meow)


    #     fitness = start1(250, 0.12, 35, 250, 20) #(250, 0.12, 35, 250, 20)
    #print(f"Лучшее решение: {best_solution}, Фитнес: {best_fitness}")
if __name__ == "__main__":
    main()