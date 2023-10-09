from symbol import Symbol
from symbol import SymbolLibrary
import numpy as np
import symbol
from actornet import ActorNet
from node import TreeDecoder
import torch
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)

def get_data(fun_name):
    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []
    # 读入训练数据集
    with open('./datasets2/trainSet/txt/' + fun_name + ".txt", "r") as f:
        fileData = f.readlines()
        for data in fileData:
            datalist = eval(data)
            train_x_list.append(datalist[:-1])
            train_y_list.append(datalist[-1])
    # 读入测试数据集
    with open('./datasets2/testSet/txt/' + fun_name + ".txt", "r") as f:
        fileData = f.readlines()
        for data in fileData:
            datalist = eval(data)
            test_x_list.append(datalist[:-1])
            test_y_list.append(datalist[-1])

    return train_x_list,train_y_list,test_x_list,test_y_list


def train(name,Epoch = 100,learning_rate = 1e-3):

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
            Symbol(symbol.protected_exp, 'exp', 1)]


    # 创建输入变量 x1,x2,...,xi
    for i in range(len(train_x_list[0])):
        symbol_list.append(Symbol(None, "x" + str(i+1), 0, x_index=i+1))

    # for item in symbol_list:
    #     print(item.name+" "+str(item.arg_num))

    symbol_set = SymbolLibrary(symbol_list)

    actorNet = ActorNet(3,32,4,symbol_set.length,symbol_set,3,device).to(device)
    optimizer = torch.optim.AdamW(actorNet.parameters(),lr=learning_rate)  # 使用Adam优化器

    obj_mse = np.inf
    obj_reward = 0

    for i in range(Epoch):
        tree_table , log_prob_list = actorNet()
        # 将树表解码为树，计算MSE、NMSE、NRMSE、Reward等值
        root = tree_table.decodeTreeTable()
        train_mse , test_mse,train_nmse ,test_nmse,train_nrmse,test_nrmse,train_reward,test_reward = TreeDecoder(root,train_x_list,train_y_list,test_x_list,test_y_list,symbol_set).calculate()
        #print("训练集MSE："+str(train_mse)+"  测试集MSE："+str(test_mse))
        # print("训练集NMSE：" + str(train_nmse) + "  测试集NMSE：" + str(test_nmse))
        # print("训练集NRMSE：" + str(train_nrmse) + "  测试集NRMSE：" + str(test_nrmse))
        #print("训练集Reward：" + str(train_reward) + "  测试集Reward：" + str(test_reward))

        if obj_mse>train_mse:
            obj_mse = train_mse
            print(" Epoch = "+str(i+1)+"   best_mse = "+str(obj_mse))
        if obj_reward<train_reward:
            obj_reward = train_reward
            print(" Epoch = " + str(i + 1) + "   best_reward = " + str(obj_reward))

        if i%1000 == 0:
            print(" Epoch = "+str(i+1)+"   mse = "+str(train_mse))
            print(" Epoch = " + str(i + 1) + "   reward = " + str(train_reward))
            # 打印网络的权重
            # for name, param in actorNet.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)

        reinforce_loss = train_reward * log_prob_list.sum(dim=0)
        optimizer.zero_grad()
        reinforce_loss.backward()
        optimizer.step()





