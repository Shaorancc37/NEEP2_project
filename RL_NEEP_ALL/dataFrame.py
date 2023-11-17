import csv
import pandas as pd
from pathlib import Path

def creatFile(path):
    # 指定要创建文件夹的路径   "../result/log/Sphere5/train"
    folder_path1 = path+"/train"
    folder_path2 = path + "/test"
    # 使用Path对象创建文件夹
    p1 = Path(folder_path1)
    p1.mkdir(parents=True, exist_ok=True)
    p2 = Path(folder_path2)
    p2.mkdir(parents=True, exist_ok=True)

    return p1 , p2




class DataFrame:
    def __init__(self,count,train_path,test_path,Train_MSE_List,Train_Best_MSE_List,Train_Reward_List,
                 Train_Best_Reward_List,Train_BestSolution,Test_MSE_List,Test_Best_MSE_List,Test_Reward_List,
                 Test_Best_Reward_List,Test_BestSolution,train_obj_solution,test_obj_solution):

        self.count = count
        self.train_path = train_path
        self.test_path = test_path
        self.Train_MSE_List = Train_MSE_List
        self.Train_Best_MSE_List = Train_Best_MSE_List
        self.Train_Reward_List = Train_Reward_List
        self.Train_Best_Reward_List = Train_Best_Reward_List
        self.Train_BestSolution = Train_BestSolution
        self.Test_MSE_List = Test_MSE_List
        self.Test_Best_MSE_List = Test_Best_MSE_List
        self.Test_Reward_List = Test_Reward_List
        self.Test_Best_Reward_List = Test_Best_Reward_List
        self.Test_BestSolution = Test_BestSolution
        self.train_obj_solution = train_obj_solution
        self.test_obj_solution = test_obj_solution



    def saveTrainData(self):
        # "../result/log/Sphere5/train/0.csv"
        data = {
            "Train_MSE_List" : self.Train_MSE_List,
            "Train_Best_MSE_List" : self.Train_Best_MSE_List,
            "Train_Reward_List" : self.Train_Reward_List,
            "Train_Best_Reward_List" : self.Train_Best_Reward_List,
            "Train_BestSolution" : self.Train_BestSolution
        }
        df = pd.DataFrame(data)
        # 将DataFrame保存为CSV文件
        path = str(self.train_path)+"/"+str(self.count)+".csv"
        df.to_csv(path, index=False)

    def saveTrainBest(self):
        s = []
        s.append(' , '.join(self.train_obj_solution))
        data = {
            "Train_Best_MSE" : self.Train_Best_MSE_List[-1],
            "Train_Best_Reward" : self.Train_Best_Reward_List[-1],
            "Solution" : s
        }
        df = pd.DataFrame(data)
        # 将DataFrame保存为CSV文件
        path = str(self.train_path) + "/" +"Train_Best_"+str(self.count)+".csv"
        df.to_csv(path, index=False)

    def saveTestData(self):
        # "../result/log/Sphere5/train/0.csv"
        data = {
            "Test_MSE_List": self.Test_MSE_List,
            "Test_Best_MSE_List": self.Test_Best_MSE_List,
            "Test_Reward_List": self.Test_Reward_List,
            "Test_Best_Reward_List": self.Test_Best_Reward_List,
            "Test_BestSolution": self.Test_BestSolution
        }
        df = pd.DataFrame(data)
        # 将DataFrame保存为CSV文件
        path = str(self.test_path) + "/" + str(self.count) + ".csv"
        df.to_csv(path, index=False)

    def saveTestBest(self):
        s = []
        s.append(' , '.join(self.test_obj_solution))
        data = {
            "Test_Best_MSE": self.Test_Best_MSE_List[-1],
            "Test_Best_Reward": self.Test_Best_Reward_List[-1],
            "Solution": s
        }
        df = pd.DataFrame(data)
        # 将DataFrame保存为CSV文件
        path = str(self.test_path) + "/" + "Test_Best_" + str(self.count) + ".csv"
        df.to_csv(path, index=False)