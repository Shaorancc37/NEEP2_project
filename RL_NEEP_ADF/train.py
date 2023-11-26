import math

import torch
import time
from pathlib import Path
import numpy as np

from RL_NEEP_ADF.ADF_Decoder import decode
from RL_NEEP_ADF.NN import Net
from RL_NEEP_ADF.dataFrame import DataFrame, creatFile

device = ("cpu")
main_char_to_idx = {
        '+': 0, '-': 1, '*': 2, '/': 3, 'sin': 4, 'cos': 5, 'ln': 6, 'e': 7,
        'ADF1': 8, 'ADF2': 9, 'ADF3': 10, 'ADF4': 11, 'ADF5': 12, 'ADF6': 13, 'ADF7': 14, 'ADF8': 15,
        'x1': 16, 'x2': 17, 'x3': 18, 'x4': 19, 'x5': 20, 'x6': 21,
        'x7': 22, 'x8': 23, 'x9': 24, 'x10': 25, 'x11': 26, 'x12': 27, 'a': 28, 'b': 29, '0': 30
    }

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


def ToExpress(main_action, main_char_list,batch_size):
    express = []
    Len = len(main_action)
    for i in range(batch_size):
        temp = []
        for j in range(Len):
            temp.append(main_char_list[main_action[j][i]])
        express.append(temp)

    return express


def train(name="",Epoch = 100,learning_rate = 1e-3,batch_size = 100,layer_num = 1, ADF_NUM=2, main_HLength = 10 , adf_HLength = 3 , cou=0):
    # 记录每个Epoch数据
    Train_MSE_List = []
    Train_Best_MSE_List = []
    Train_Reward_List = []
    Train_Best_Reward_List = []
    Train_BestSolution = []
    Train_Best_ADF = []
    Test_MSE_List = []
    Test_Best_MSE_List = []
    Test_Reward_List = []
    Test_Best_Reward_List = []
    Test_BestSolution = []
    Test_Best_ADF = []


    main_input_size = main_HLength*2 + 1
    adf_input_size = adf_HLength*2 +1
    updateSeed(cou)
    print("实验 ： "+str(name)+" 开始")
    # 获取训练集 和 测试集数据
    train_x_list, train_y_list, test_x_list, test_y_list = get_data(name)
    data_length = len(train_x_list)

    teri_num = len(train_x_list[0])
    main_char_list = ['+', '-', '*', '/', 'sin', 'cos', 'ln', 'e']
    adf_char_list = ['+', '-', '*', '/', 'sin', 'cos', 'ln', 'e', 'a', 'b']
    for i in range(ADF_NUM):
        char = "ADF" + str(i + 1)
        main_char_list.append(char)
    for i in range(teri_num):
        char = "x" + str(i + 1)
        main_char_list.append(char)
    main_fun_length = ADF_NUM + 8
    adf_fun_length = 8

    main_nn = Net(batch_size, main_input_size, 32, layer_num, len(main_char_list), main_fun_length, main_char_list, main_char_to_idx, device)
    optimizer_main = torch.optim.AdamW(main_nn.parameters(), lr=learning_rate)  # 使用Adam优化器
    ADF_NN = []
    adf_opt_list = []
    for i in range(ADF_NUM):
        ADF_NN.append(Net(batch_size, adf_input_size, 32, layer_num, len(adf_char_list), adf_fun_length, adf_char_list, main_char_to_idx, device))
        adf_opt_list.append(torch.optim.AdamW(ADF_NN[i].parameters(), lr=learning_rate))

    train_obj_mse = np.inf
    train_obj_reward = 0
    train_obj_solution = None
    train_obj_main = None
    train_obj_adf = []
    test_obj_mse = np.inf
    test_obj_reward = 0
    test_obj_solution = None
    test_obj_main = None
    test_obj_adf = []

    for i in range(Epoch):
        main_action, main_log_prob = main_nn()
        main_express = ToExpress(main_action,main_char_list,batch_size)
        adf_express = []
        adf_log_prob_list = []
        for j in range(ADF_NUM):
            adf_action ,adf_log_prob = ADF_NN[j]()
            adf_log_prob_list.append(adf_log_prob)
            adf_express.append(ToExpress(adf_action,adf_char_list,batch_size))


        train_mse_list = []
        test_mse_list = []
        train_reward_list = []
        test_reward_list = []
        train_nrmse_list = []
        test_nrmse_list = []

        # 计算一个Batch的解的数据
        for j in range(batch_size):



            main_decoder = decode(main_express[j],len(main_express[j]))
            adf_decoder_list = []
            for k in range(ADF_NUM):
                adf_decoder_list.append(decode(adf_express[k][j],len(adf_express[k][j])))
            main_root = main_decoder.star()
            adf_root = []
            for k in range(ADF_NUM):
                adf_root.append(adf_decoder_list[k].star())
            root = main_decoder.reductionToADF(main_root,adf_root)
            # main_decoder.display(root)
            # input()


            train_MSE = 0
            test_MSE = 0
            train_true_y = 0
            test_true_y = 0

            #计算预测值
            for k in range(data_length):
                # 训练集上计算
                train_predict_value = main_decoder.calculateFitness(root , train_x_list[k])
                train_true_y += train_y_list[k]
                try:
                    train_MSE += (train_predict_value - train_y_list[k]) ** 2
                except:
                    train_MSE += math.fabs(train_predict_value - train_y_list[k])
                # 测试集上计算
                test_predict_value = main_decoder.calculateFitness(root, test_x_list[k])
                test_true_y += test_y_list[k]
                try:
                    test_MSE += (test_predict_value - test_y_list[k]) ** 2
                except:
                    test_MSE += math.fabs(test_predict_value - test_y_list[k])

            train_MSE = round(train_MSE/data_length , 6)
            test_MSE = round(test_MSE/data_length , 6)
            train_NRMSE = math.sqrt(train_MSE) / (train_true_y / data_length)
            test_NRMSE = math.sqrt(test_MSE) / (test_true_y / data_length)
            train_reward = 1 / (1 + train_NRMSE)
            test_reward = 1 / (1 + test_NRMSE)
            train_mse_list.append(train_MSE)
            test_mse_list.append(test_MSE)
            train_nrmse_list.append(train_NRMSE)
            test_nrmse_list.append(test_NRMSE)
            train_reward_list.append(train_reward)
            test_reward_list.append(test_reward)

        # 得到这一个Epoch中，整个Batch中的最优值,并更新历史最优
        train_min_mse = min(train_mse_list)
        train_max_reward = max(train_reward_list)
        test_min_mse = min(test_mse_list)
        test_max_reward = max(test_reward_list)

        # 更新全局最优解
        if train_obj_mse > train_min_mse:
            train_obj_mse = train_min_mse
            train_obj_solution = main_express[train_mse_list.index(train_min_mse)]
            temp = []
            for j in range(ADF_NUM):
                temp.append(adf_express[j][train_mse_list.index(train_min_mse)])
            train_obj_adf = temp
        if train_obj_reward < train_max_reward:
            train_obj_reward = train_max_reward
        if test_obj_mse > test_min_mse:
            test_obj_mse = test_min_mse
            test_obj_solution = main_express[test_mse_list.index(test_min_mse)]
            temp = []
            for j in range(ADF_NUM):
                temp.append(adf_express[j][test_mse_list.index(test_min_mse)])
            test_obj_adf = temp
        if test_obj_reward < test_max_reward:
            test_obj_reward = test_max_reward

        Train_MSE_List.append(train_min_mse)
        Train_Best_MSE_List.append(train_obj_mse)
        Train_Reward_List.append(train_max_reward)
        Train_Best_Reward_List.append(train_obj_reward)
        Train_BestSolution.append(main_express[train_mse_list.index(train_min_mse)])
        temp_train_adf = []
        for j in range(ADF_NUM):
            temp_train_adf.append(adf_express[j][train_mse_list.index(train_min_mse)])
        Train_Best_ADF.append(temp_train_adf)
        Test_MSE_List.append(test_min_mse)
        Test_Best_MSE_List.append(test_obj_mse)
        Test_Reward_List.append(test_max_reward)
        Test_Best_Reward_List.append(test_obj_reward)
        Test_BestSolution.append(main_express[test_mse_list.index(test_min_mse)])
        temp_test_adf = []
        for j in range(ADF_NUM):
            temp_test_adf.append(adf_express[j][test_mse_list.index(test_min_mse)])
        Test_Best_ADF.append(temp_test_adf)


        reward = torch.tensor(train_reward_list)
        baseline = torch.sum(reward) / batch_size

        # 更新网络
        main_loss = torch.sum((reward - baseline) * main_log_prob.sum(dim=0)) / batch_size
        main_loss = -1 * (main_loss)
        optimizer_main.zero_grad()
        main_loss.backward()
        optimizer_main.step()

        for j in range(ADF_NUM):
            adf_loss = torch.sum((reward - baseline) * adf_log_prob_list[j].sum(dim=0)) / batch_size
            adf_loss = -1 * (adf_loss)
            adf_opt_list[j].zero_grad()
            adf_loss.backward()
            adf_opt_list[j].step()

    # 创建存放训练集和测试集的文件夹
    path = "../result/log/" + name
    train_path, test_path = creatFile(path)

    # 写入文件
    DateLog = DataFrame(cou,
                        train_path,
                        test_path,
                        Train_MSE_List,
                        Train_Best_MSE_List,
                        Train_Reward_List,
                        Train_Best_Reward_List,
                        Train_BestSolution,
                        Train_Best_ADF,
                        Test_MSE_List,
                        Test_Best_MSE_List,
                        Test_Reward_List,
                        Test_Best_Reward_List,
                        Test_BestSolution,
                        Test_Best_ADF,
                        train_obj_solution,
                        test_obj_solution,
                        train_obj_adf,
                        test_obj_adf
                        )
    DateLog.saveTrainData()
    DateLog.saveTrainBest()
    DateLog.saveTestData()
    DateLog.saveTestBest()
    print("保存完毕")








