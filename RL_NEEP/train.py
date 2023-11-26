from RL_NEEP.Net import Net
from RL_NEEP.decoder import Decoder
from RL_NEEP_ALL import symbol
from RL_NEEP_ALL.dataFrame import DataFrame, creatFile
from RL_NEEP_ALL.symbol import Symbol
from RL_NEEP_ALL.symbol import SymbolLibrary
import numpy as np

from RL_NEEP_ALL.node import TreeDecoder
import torch
import time
from pathlib import Path

device = ("cpu")

def get_data(fun_name):
    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []
    # 读入训练数据集
    with open('../datasets2/trainSet/txt/' + fun_name + ".txt", "r") as f:
        fileData = f.readlines()
        for data in fileData:
            datalist = eval(data)
            train_x_list.append(datalist[:-1])
            train_y_list.append(datalist[-1])
    # 读入测试数据集
    with open('../datasets2/testSet/txt/' + fun_name + ".txt", "r") as f:
        fileData = f.readlines()
        for data in fileData:
            datalist = eval(data)
            test_x_list.append(datalist[:-1])
            test_y_list.append(datalist[-1])

    return train_x_list,train_y_list,test_x_list,test_y_list


def updateSeed(cou):
    torch.manual_seed(cou)
    torch.cuda.manual_seed_all(cou)
    np.random.seed(cou)


def train(name="",Epoch = 100,learning_rate = 1e-3,batch_size = 100,layer_num = 1,cou=0):
    updateSeed(cou)
    print("实验 ： "+str(name)+" 开始")
    # 记录每个Epoch数据
    Train_MSE_List = []
    Train_Best_MSE_List = []
    Train_Reward_List = []
    Train_Best_Reward_List = []
    Train_BestSolution = []
    Test_MSE_List = []
    Test_Best_MSE_List = []
    Test_Reward_List = []
    Test_Best_Reward_List = []
    Test_BestSolution = []

    # 获取训练集 和 测试集数据
    train_x_list, train_y_list, test_x_list, test_y_list = get_data(name)
    # 创建符号集
    symbol_list = None
    if symbol_list is None:
        symbol_list = [
            Symbol(np.add, '+', 2),
            Symbol(np.subtract, '-', 2),
            Symbol(np.multiply, '*', 2),
            Symbol(symbol.protected_div, '/', 2),
            Symbol(np.sin, 'sin', 1),
            Symbol(np.cos, 'cos', 1),
            Symbol(symbol.protected_log, 'log', 1),
            Symbol(symbol.protected_exp, 'exp', 1)
        ]

    # 创建输入变量 x1,x2,...,xi
    for i in range(len(train_x_list[0])):
        symbol_list.append(Symbol(None, "x" + str(i+1), 0, x_index=i+1))

    # for item in symbol_list:
    #     print(item.name+" "+str(item.arg_num))

    symbol_set = SymbolLibrary(symbol_list)

    actorNet = Net(batch_size,32,32,layer_num,symbol_set.length,symbol_set,31,device).to(device)
    optimizer = torch.optim.AdamW(actorNet.parameters(),lr=learning_rate)  # 使用Adam优化器

    train_obj_mse = np.inf
    train_obj_reward = 0
    train_obj_solution = None
    test_obj_mse = np.inf
    test_obj_reward = 0
    test_obj_solution = None

    for i in range(Epoch):
        action , log_prob1_list = actorNet()
        print(action)
        input()

        root = []
        train_mse_list = []
        train_nmse_list = []
        train_nrmse_list = []
        train_reward_list = []
        test_mse_list = []
        test_nmse_list = []
        test_nrmse_list = []
        test_reward_list = []
        for j in range(batch_size):
            root.append(Decoder(j,action,symbol_set).creat())
            train_mse , test_mse,train_nmse ,test_nmse,train_nrmse,test_nrmse,train_reward,test_reward = TreeDecoder(root[j],train_x_list,train_y_list,test_x_list,test_y_list,symbol_set).calculate()
            # 将值加到列表里
            train_mse_list.append(train_mse)
            train_nmse_list.append(train_nmse)
            train_nrmse_list.append(train_nrmse)
            train_reward_list.append(train_reward)
            test_mse_list.append(test_mse)
            test_nmse_list.append(test_nmse)
            test_nrmse_list.append(test_nrmse)
            test_reward_list.append(test_reward)

        # 得到这一个Epoch中，整个Batch中的最优值,并更新历史最优
        train_min_mse = min(train_mse_list)
        train_max_reward = max(train_reward_list)
        test_min_mse = min(test_mse_list)
        test_max_reward = max(test_reward_list)

        # 更新全局最优解
        if train_obj_mse>train_min_mse:
            train_obj_mse = train_min_mse
            train_obj_solution = Decoder(train_mse_list.index(train_min_mse),action,symbol_set).getString()#tree_table[train_mse_list.index(train_min_mse)].getSolution()
        if train_obj_reward<train_max_reward:
            train_obj_reward = train_max_reward
        if test_obj_mse>test_min_mse:
            test_obj_mse = test_min_mse
            test_obj_solution = Decoder(test_mse_list.index(test_min_mse),action,symbol_set).getString()#tree_table[test_mse_list.index(test_min_mse)].getSolution()
        if test_obj_reward<test_max_reward:
            test_obj_reward = test_max_reward

        Train_MSE_List.append(train_min_mse)
        Train_Best_MSE_List.append(train_obj_mse)
        Train_Reward_List.append(train_max_reward)
        Train_Best_Reward_List.append(train_obj_reward)
        Train_BestSolution.append(Decoder(train_mse_list.index(train_min_mse),action,symbol_set).getString())
        Test_MSE_List.append(test_min_mse)
        Test_Best_MSE_List.append(test_obj_mse)
        Test_Reward_List.append(test_max_reward)
        Test_Best_Reward_List.append(test_obj_reward)
        Test_BestSolution.append(Decoder(test_mse_list.index(test_min_mse),action,symbol_set).getString())


        # print(" Epoch = "+str(i+1)+"  best_mse = "+str(obj_mse))
        # print(" Epoch = " + str(i + 1)+ "  best_reward = " + str(obj_reward))
        reward = torch.tensor(train_reward_list)
        baseline = torch.sum(reward)/batch_size

        # 更新 符号概率 途径网络
        reinforce_loss1 = torch.sum((reward - baseline) * log_prob1_list.sum(dim=0)) / batch_size
        loss = -1*(reinforce_loss1 )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 创建存放训练集和测试集的文件夹
    path = "../result/log/"+name
    train_path,test_path = creatFile(path)

    # 写入文件
    DateLog = DataFrame(cou,
                        train_path,
                        test_path,
                        Train_MSE_List,
                        Train_Best_MSE_List,
                        Train_Reward_List,
                        Train_Best_Reward_List,
                        Train_BestSolution,
                        Test_MSE_List,
                        Test_Best_MSE_List,
                        Test_Reward_List,
                        Test_Best_Reward_List,
                        Test_BestSolution,
                        train_obj_solution,
                        test_obj_solution
                        )
    DateLog.saveTrainData()
    DateLog.saveTrainBest()
    DateLog.saveTestData()
    DateLog.saveTestBest()
    print("保存完毕")


if __name__ == '__main__':
    train(name="Sphere5", Epoch=100, learning_rate=1e-3, batch_size=10, layer_num=1, cou=0)

