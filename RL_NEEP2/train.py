from symbol import Symbol
from symbol import SymbolLibrary
import numpy as np
import symbol
from actornet import ActorNet
from node import TreeDecoder
import torch
import time
# 启用异常检测
torch.autograd.set_detect_anomaly(True)


device = ("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)

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


def train(name,Epoch = 100,learning_rate = 1e-3,batch_size = 100):

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

    actorNet = ActorNet(batch_size,32,32,1,symbol_set.length,symbol_set,3,device).to(device)
    optimizer = torch.optim.AdamW(actorNet.parameters(),lr=learning_rate)  # 使用Adam优化器

    obj_mse = np.inf
    obj_reward = 0

    for i in range(Epoch):
        # 记录开始时间
        start_time = time.time()
        tree_table , log_prob1_list, log_prob2_list = actorNet()
        # TODO 将一个Batch的树表解码为树，分别在 训练集 和 测试集 上计算MSE、NMSE、NRMSE、Reward等值
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
            root.append(tree_table[j].decodeTreeTable())
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
        min_mse = min(train_mse_list)
        max_reward = max(train_reward_list)
        if obj_mse>min_mse:
            obj_mse = min_mse
        if obj_reward<max_reward:
            obj_reward = max_reward

        print(" Epoch = "+str(i+1)+" min_mse = "+str(min_mse)+"  best_mse = "+str(obj_mse))
        print(" Epoch = " + str(i + 1) + " max_reward = " + str(max_reward) + "  best_reward = " + str(obj_reward))


        reward = torch.tensor(train_reward_list)
        baseline = torch.sum(reward)/batch_size

        # 更新 符号概率 途径网络
        reinforce_loss1 = torch.sum((reward - baseline) * log_prob1_list.sum(dim=0)) / batch_size
        reinforce_loss2 = torch.sum((reward - baseline) * log_prob2_list.sum(dim=0)) / batch_size
        loss = reinforce_loss1 + reinforce_loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新 位置概率 途径网络
        # reinforce_loss2 = torch.sum((reward - baseline) * log_prob2_list.sum(dim=0)) / batch_size
        # optimizer.zero_grad()
        # reinforce_loss2.backward()
        # optimizer.step()

        # 记录结束时间
        end_time = time.time()
        # 计算代码执行时间
        execution_time = end_time - start_time
        # 打印执行时间
        print("执行时间:", execution_time, "秒")






