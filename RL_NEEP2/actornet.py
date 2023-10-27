
import torch
import torch.nn as nn
import torch.nn.functional as F

from lstm_subfun import updateTreeTable, getInput ,updateFinished
from treetable import TreeTable , Row
import torch.distributions as dist
from torch.distributions import Categorical

class GaussianActivation(nn.Module):
    def forward(self, x):
        # 使用高斯函数作为激活函数
        return torch.exp(-x**2)


def getSpecificNode(tree_table, specific_node_index):
    # 从上到下查树表，看哪一行的左右父有空缺，这行就是特定节点
    for i in range(specific_node_index,tree_table.length):
        temp_row = tree_table.rows[i]
        if temp_row.left_pos is None:
            return i
        elif temp_row.right_pos is None:
            return i
        elif temp_row.father_pos is None:
            return i


def judge(finished):
    for item in finished:
        if item == 0:
            return True
    return False


def updateByFinished(log_prob1, finished,batch_size):
    for i in range(batch_size):
        if finished[i] == 1:
            log_prob1[i] = 0
    return log_prob1


class ActorNet(nn.Module):
    def __init__(self,batch_size,input_size, hidden_size, num_layers, output_size, symbol_set , max_height, device):
        super(ActorNet, self).__init__()
        # 定义输入的embedding层
        self.input_embedding_layer = nn.Embedding(2,8)
        self.input_layer = nn.Linear(3*8,input_size)

        # 定义LSTM网络结构
        self.lstm = nn.LSTM(input_size,hidden_size, num_layers, batch_first=True)
        # 添加BatchNormalization层
        self.bn = nn.BatchNorm1d(hidden_size)
        # 定义输出层
            # fc_output是符号对应的输出
        self.fc_output = nn.Linear(hidden_size,output_size)
            # fc_position是当生成父节点时，判断特定节点对应父节点位置的输出
        self.fc_position = nn.Linear(hidden_size,3)
        # 高斯激活函数
        self.gaussian_activation = GaussianActivation()
        self.batch_size = batch_size
        self.symbol_set = symbol_set
        self.max_height = max_height
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

    def get_lstm_input(self,lstm_input):
        input_embedding = self.input_embedding_layer(lstm_input).reshape(self.batch_size,-1)
        return input_embedding

    def forward(self):

        # 初始化lstm的隐藏层输出 和 细胞状态输出
        h_out = torch.rand((self.num_layers, self.hidden_size), dtype=torch.float32).to(self.device)
        c_out = torch.rand((self.num_layers, self.hidden_size), dtype=torch.float32).to(self.device)
        # 初始化树表
        tree_table = []
        for i in range(self.batch_size):
            tree_table.append(TreeTable())
        # 初始化log_p分布
        log_prob1_list = []
        log_prob2_list = []
        # 初始化特定节点
        specific_node_index = []
        for i in range(self.batch_size):
            specific_node_index.append(None)

        # 初始化finished_list ,用于记录一个Batch的树表各自的完成情况
        finished = torch.zeros(self.batch_size,dtype = torch.int32)
        #print("finishen : "+str(finished))


        # 判断是否有树还没生长完
        while judge(finished) :
        # 获取输入
            lstm_input , lstm_type = getInput(tree_table,specific_node_index,self.batch_size)
        # 送入网络获得输出
            x = self.get_lstm_input(lstm_input)
            x = self.input_layer(x.to(torch.float32))
            output, (h_out, c_out) = self.lstm(x)
            # 进行batchnorm的计算
            output = self.bn(output)
            # TODO 这部分是计算 符号概率 的输出
            output_symbol = self.fc_output(self.gaussian_activation(output))
            # 得到符号的概率输出
            output_symbol = F.softmax(output_symbol, dim=1)
            # 建立分布
            dist1 = Categorical(output_symbol)
            # 采样
            action1 = dist1.sample()
            # TODO 这部分是计算 位置概率 的输出
            output_pos = self.fc_position(self.gaussian_activation(output))
            # 得到位置概率的输出
            output_pos = F.softmax(output_pos, dim=1)
            # 建立分布
            dist2 = Categorical(output_pos)
            # 采样
            action2 = dist2.sample()
            # TODO 接下来要填表，将一个Batch的树表填好，填完返回特定节点
            # 填表 : symbol_set, action, tree_table , specific_node_index
            tree_table, specific_node_index,action1 = updateTreeTable(lstm_type ,self.symbol_set,output_symbol, action1,action2, tree_table, specific_node_index,self.batch_size,self.max_height,finished)
            # TODO 计算log_prob
            # 计算符号生成的log_p
            log_prob1 = dist1.log_prob(action1)
            log_prob1 = updateByFinished(log_prob1, finished, self.batch_size)
            log_prob1_list.append(log_prob1)
            # 计算位置选取的log_p
            log_prob2 = dist2.log_prob(action2)
            log_prob2 = updateByFinished(log_prob2, finished, self.batch_size)
            log_prob2_list.append(log_prob2)

            # 每次生长完一个节点后都计算一下树的高度
            for i in range(self.batch_size):
                tree_table[i].updateHeight()
                #print(tree_table[i].height)
            # 生长完一个节点应该更新一遍finished列表
            finished = updateFinished(tree_table,finished,self.batch_size)


        return tree_table , torch.stack(log_prob1_list) , torch.stack(log_prob2_list)
