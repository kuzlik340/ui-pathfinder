from project import start1
import random


def main():
        # for s in range (3,20):
        #     for m in range (2, 10):
        #for i in range(100):
                meow = 0
                print("\n")
                #print("T E S T I N G   W I T H   elite size = ", m, " and tournament size = ", s)
                for i in range(100):
                    meow += start1(500, 0.12, 45, 200, 30) #14
                    if i % 10 == 0:
                        print(i)
                print("\nTotal ideal attempts = ", meow, "out of 100")
                # if meow >= 70:
                #     break
        #start(200, 0.14, 2, 100, 10)
#meow += start(200, 0.139, 3, 130, 10) #14 best config 40 out of 100






if __name__ == "__main__":
    main()