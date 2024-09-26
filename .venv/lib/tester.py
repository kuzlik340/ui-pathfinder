from project import start
import random


def main():
        print("\n")
        print("T E S T I N G   W I T H")
        meow = 0
        for i in range(50):
          meow += start(250, 0.14, 30, 200, 10) #14
        print(1, ": Total ideal attempts = ", meow, "out of 50")
        #start(100, 0.14, 14, 100, 10)







if __name__ == "__main__":
    main()