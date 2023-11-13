import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as dist
from torch.distributions import Categorical

from RL_NEEP2 import symbol
from RL_NEEP2.symbol import Symbol, SymbolLibrary


class GaussianActivation(nn.Module):
    def forward(self, x):
        # 使用高斯函数作为激活函数
        return torch.exp(-x**2)


class Net(nn.Module):

    def __init__(self,batch_size,input_size, hidden_size,num_layers,output_size, symbol_set , max_length, device):
        super(Net,self).__init__()

        # 定义输入的embedding层
        self.input_embedding_layer = nn.Embedding(2, 8)
        self.input_layer = nn.Linear(3 * 8, input_size)

        # 定义LSTM网络结构
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 添加BatchNormalization层
        self.bn = nn.BatchNorm1d(hidden_size)
        # 定义输出层
        # fc_output是符号对应的输出
        self.fc_output = nn.Linear(hidden_size, output_size)
        # 高斯激活函数
        self.gaussian_activation = GaussianActivation()
        self.batch_size = batch_size
        self.symbol_set = symbol_set
        self.max_length = max_length
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.head_length = (max_length - 1)/2
        self.tail_length = self.head_length + 1


    def forward(self):

        # 初始化lstm的隐藏层输出 和 细胞状态输出
        h_out = torch.rand((self.num_layers, self.hidden_size), dtype=torch.float32).to(self.device)
        c_out = torch.rand((self.num_layers, self.hidden_size), dtype=torch.float32).to(self.device)
        input = torch.ones(self.batch_size, self.input_size)

        # 初始化 结果列表
        res = []
        # for i in range(self.batch_size):
        #     res.append([])

        # 初始化log_p分布
        log_prob_list = []


        for i in range(self.max_length):
            # 前向传播
            output,(h_out, c_out) = self.lstm(input, (h_out, c_out))
            # 进行batchnorm的计算
            output = self.bn(output)
            # TODO 这部分是计算 符号概率 的输出
            output_symbol = self.fc_output(self.gaussian_activation(output))

            # 得到符号的概率输出
            output_symbol1 = F.softmax(output_symbol, dim=1)
            # 建立分布
            dist = Categorical(output_symbol1)
            # 采样
            action = dist.sample()

            if i < self.head_length:
                res.append(action)
            else:
                temp_tensor = output_symbol[:, self.symbol_set.input_symbols[0]:]
                # 得到符号的概率输出
                output_symbol1 = F.softmax(temp_tensor, dim=1)
                # 建立分布
                dist1 = Categorical(output_symbol1)
                # 采样
                action = dist1.sample()
                action = action + torch.tensor(self.symbol_set.input_symbols[0])
                res.append(action)
            # 计算符号生成的log_p
            log_prob = dist.log_prob(action)
            log_prob_list.append(log_prob)

        #print(res)
        # print(log_prob_list)


        return res , torch.stack(log_prob_list)




if __name__ == '__main__':
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
    for i in range(10):
        symbol_list.append(Symbol(None, "x" + str(i + 1), 0, x_index=i + 1))

    # for item in symbol_list:
    #     print(item.name+" "+str(item.arg_num))

    symbol_set = SymbolLibrary(symbol_list)
    print(symbol_set.arg_nums)
    print(symbol_set.input_symbols)




    net = Net(10,32,32,1,symbol_set.length,symbol_set,31,None)
    net()