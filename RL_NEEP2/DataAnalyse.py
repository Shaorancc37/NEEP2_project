import csv
import numpy as np
import pandas as pd


def getNEEP2Analyse(problem):
    ans = []
    for i in range(30):
        path = "../result/log/"+str(problem)+"/test/Test_Best_"+str(i)+".csv"
        temp = []
        with open(path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                temp = row
        ans.append(temp[0])
    return np.median(ans)


if __name__ == '__main__':
    problem = "Sphere5"
    # 分析NEEP230次实验的结果，得到每个实验的最好结果中位数
    getNEEP2Analyse(problem)