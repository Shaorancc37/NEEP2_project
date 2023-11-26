import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as dist
from torch.distributions import Categorical

from RL_NEEP_ALL import symbol
from RL_NEEP_ALL.symbol import Symbol, SymbolLibrary


device = ("cpu")

class GaussianActivation(nn.Module):
    def forward(self, x):
        # 使用高斯函数作为激活函数
        return torch.exp(-x**2)


# 计算位置向量编码
def positional_encoding(seq_length, embedding_dim):
    encoding = torch.zeros(seq_length, embedding_dim)
    for pos in range(seq_length):
        for i in range(embedding_dim):
            if i % 2 == 0:
                encoding[pos, i] = np.sin(torch.tensor(pos / (10000 ** (i / embedding_dim))))
            else:
                encoding[pos, i] = np.cos(torch.tensor(pos / (10000 ** ((i - 1) / embedding_dim))))
    return encoding


class Net(nn.Module):

# input_size : mian\adf的表达式长度
# output_size : main\adf对应的符号集长度
#
    def __init__(self,batch_size,input_size, hidden_size,num_layers,output_size,fun_length, main_char_list,char_to_idx, device):
        super(Net,self).__init__()

        # 定义输入的embedding层
        # self.input_embedding_layer = nn.Embedding(2, 8)
        self.position_vector = positional_encoding(input_size, 3 * input_size).to(device)
        self.main_char_list = main_char_list
        self.char_to_idx = char_to_idx
        self.embedings = nn.Embedding(len(char_to_idx), 3, padding_idx=char_to_idx["0"]).to(device)
        self.input_layer = nn.Linear(3 * input_size, input_size)

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
        self.device = device
        self.num_layers = num_layers
        self.fun_length = fun_length
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.max_length = input_size
        self.head_length = (input_size - 1) / 2
        self.tail_length = self.head_length + 1




    def forward(self):
        express_list = []
        express = []
        for i in range(self.input_size):
            express.append("0")
        for i in range(self.batch_size):
            express_list.append(copy.deepcopy(express))

        # 初始化lstm的隐藏层输出 和 细胞状态输出
        h_out = torch.rand((self.num_layers, self.hidden_size), dtype=torch.float32).to(self.device)
        c_out = torch.rand((self.num_layers, self.hidden_size), dtype=torch.float32).to(self.device)

        # 初始化 结果列表
        res = []
        # for i in range(self.batch_size):
        #     res.append([])
        # 初始化log_p分布
        log_prob_list = []


        for i in range(self.max_length):
            # 将state通过embedding层将离散字符串转化为连续向量（输入的预处理）
            x = self.inputPretreatment(express_list)
            # 将词向量拼起来
            x = x.view(self.batch_size, -1)
            x = x+ self.position_vector[i]
            x = self.input_layer(x)

            # 前向传播
            output,(h_out, c_out) = self.lstm(x, (h_out, c_out))
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
                temp_tensor = output_symbol[:,self.fun_length:]
                # 得到符号的概率输出
                output_symbol1 = F.softmax(temp_tensor, dim=1)
                # 建立分布
                dist1 = Categorical(output_symbol1)
                # 采样
                action = dist1.sample()
                action = action + torch.tensor(self.fun_length)
                res.append(action)


            # 计算符号生成的log_p
            log_prob = dist.log_prob(action)
            log_prob_list.append(log_prob)

        # print(res)
        # input()
        # print(log_prob_list)


        return res , torch.stack(log_prob_list)


    def inputPretreatment(self,express_list):
        indexed_data = []
        for express in express_list:
            char_index = []
            for char in express:
                char_index.append(self.char_to_idx[char])
            indexed_data.append(char_index)
        # 将索引列表转换为 PyTorch Tensor
        indexed_data_tensor = torch.tensor(indexed_data).to(self.device)
        # 将输入张量传递给Embedding层，得到对应的向量表示
        embedded = self.embedings(indexed_data_tensor)
        return embedded

# ,batch_size,input_size, hidden_size,num_layers,output_size, char_to_idx, device
if __name__ == '__main__':
    batch_size = 10
    ADF_NUM = 6
    teri_num = 5
    main_char_list = ['+','-','*','/','sin', 'cos', 'ln', 'e']
    adf_char_list = ['+','-','*','/','sin', 'cos', 'ln', 'e', 'a', 'b']
    for i in range(ADF_NUM):
        char = "ADF"+str(i+1)
        main_char_list.append(char)
    for i in range(teri_num):
        char = "x"+str(i+1)
        main_char_list.append(char)
    main_fun_length = ADF_NUM + 8
    adf_fun_length = 8
    main_char_to_idx = {
        '+': 0, '-': 1, '*': 2, '/': 3, 'sin': 4, 'cos': 5, 'ln': 6, 'e': 7,
        'ADF1': 8, 'ADF2': 9, 'ADF3': 10, 'ADF4': 11, 'ADF5': 12, 'ADF6': 13, 'ADF7': 14, 'ADF8': 15,
        'x1': 16, 'x2': 17, 'x3': 18, 'x4': 19, 'x5': 20, 'x6': 21,
        'x7': 22, 'x8': 23, 'x9': 24, 'x10': 25, 'x11': 26, 'x12': 27, 'a': 28, 'b': 29, '0': 30
    }
    main_nn = Net(batch_size,21,32,1,len(main_char_list),main_fun_length,main_char_list,main_char_to_idx,device)
    adf1_nn = Net(batch_size,7,32,1,len(adf_char_list),adf_fun_length,adf_char_list,main_char_to_idx,device)
    main_action , main_log_prob_list = main_nn()
    # action大小为 序列长度 * batch_size
    adf1_action , adf1_log_prob_list = adf1_nn()




