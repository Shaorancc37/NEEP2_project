import math

import numpy as np


class Node:

    def __init__(self,symbol=None):
        self.symbol = symbol
        self.left_child = None
        self.right_child = None


def cal(root, train_x,symbol_set):
    # 终止节点直接返回
    if root.symbol.arg_num == 0:
        index = int(root.symbol.name[1:]) - 1
        return train_x[index]
    elif root.symbol.arg_num == 1: # 一元操作符log,exp,sin,cos
        var1 = cal(root.left_child,train_x,symbol_set)
        if root.symbol.name == "sin":
            if math.isinf(var1) is True:
                return 1
            else:
                return root.symbol.function(var1)
        elif root.symbol.name == "cos":
            if math.isinf(var1) is True:
                return 1
            else:
                return root.symbol.function(var1)
        return root.symbol.function(var1)
    elif root.symbol.arg_num == 2:  # 二元操作符+，-，*，/
        var1 = cal(root.left_child, train_x, symbol_set)
        var2 = cal(root.right_child, train_x, symbol_set)
        return root.symbol.function(var1,var2)


class TreeDecoder:
    def __init__(self,root,train_x,train_y,test_x,test_y,symbol_set):
        self.root = root
        self.mse = None
        self.nmse = None
        self.nrmse = None
        self.reward = None
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.symbol_set = symbol_set

    def calculate(self):
        train_mse = 0
        test_mse = 0
        train_nmse = 0
        test_nmse = 0
        train_nrmse = 0
        test_nrmse = 0
        train_reward = 0
        test_reward = 0
        trainy_true = 0
        testy_true = 0
        trainy_std = np.std(self.train_y)
        testy_std = np.std(self.test_y)
        length = len(self.train_x)
        for i in range(length):
            # try:
            predict_train = cal(self.root,self.train_x[i],self.symbol_set)
            #print("predict_train = "+str(predict_train))
            predict_test = cal(self.root,self.test_x[i],self.symbol_set)
            train_mse += (predict_train - self.train_y[i]) ** 2
            test_mse += (predict_test - self.test_y[i]) ** 2
            trainy_true += self.train_y[i]
            testy_true += self.test_y[i]
            # except:
            #     print("calculate()计算错误 node 65rows")
        # 计算
        train_mse = round(train_mse/length,6)
        test_mse = round(test_mse/length,6)
        train_nmse = train_mse/(trainy_std**2)
        test_nmse = test_mse/(testy_std**2)
        train_nrmse = math.sqrt(train_mse) / (trainy_true / length)
        test_nrmse = math.sqrt(test_mse) / (testy_true / length)
        train_reward = 1 / (1 + train_nrmse)
        test_reward = 1 / (1 + test_nrmse)

        return train_mse,test_mse,train_nmse,test_nmse,train_nrmse,test_nrmse,train_reward,test_reward


            


