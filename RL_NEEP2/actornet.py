import functorch.dim
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
            #print("生成+1")
        # 获取输入
            lstm_input , lstm_type = getInput(tree_table,specific_node_index,self.batch_size)
        # 送入网络获得输出
            x = self.get_lstm_input(lstm_input)
            x = self.input_layer(x.to(torch.float32))
            output, (h_out, c_out) = self.lstm(x)
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







'''
        # 开始构造树 TODO:此处循环的控制需要重新写
        while 1: #tree_table.judge() and tree_table.height<=self.max_height+1:
                lstm_input = torch.zeros(self.batch_size,3,dtype=torch.int64).to(self.device)
                print("lstm_input.shape = "+str(lstm_input.shape))
                # 将one-hot转换为embedding
                x = self.get_lstm_input(lstm_input)
                x = self.input_layer(x.to(torch.float32))
                output,(h_out,c_out) = self.lstm(x)
                # 符号概率的输出
                output_symbol = self.fc_output(self.gaussian_activation(output))
                # 得到符号的概率输出
                output_symbol = F.softmax(output_symbol,dim=1)
                # 建立分布
                dist = Categorical(output_symbol)
                # 采样
                action = dist.sample()
                # 计算log_p
                log_prob = dist.log_prob(action)
                log_prob_list.append(log_prob)
                # TODO:这里需要将一个Batch中的树表分别填写好
                # 填表 : type(首个符号1、正常生长符号2、限高符号3), symbol_set, action, tree_table , specific_node_index
                tree_table , specific_node_index = updateTreeTable(1,self.symbol_set,action,tree_table,specific_node_index,self.batch_size)

                # 在symbol_set中找到符号，填表
                # temp_symbol = self.symbol_set.symbol_list[action]
                # #print("输出："+temp_symbol.name) 2
                # # 创建树表中的一行数据row,填序号，符号，左右节点和节点类型
                # new_row = Row(tree_table.length)
                # new_row.symbol = temp_symbol
                # new_row.arg_num = temp_symbol.arg_num
                # if new_row.arg_num ==0 : #终止节点
                #     new_row.left_pos = -1
                #     new_row.right_pos = -1
                # elif new_row.arg_num == 1 : # 单操作符节点
                #     new_row.right_pos = -1
                # tree_table.addRow(new_row)
                # # print("当前表：") 3
                # # tree_table.display()
                # # 更新特定节点
                # specific_node_index = 0
                # 是否需要计算树高？后续再看
            else :
                # 当前树高度已经达到max_height ，向上无法生长节点，直接判定为根节点，向下只能生长终止节点
                if tree_table.height >= self.max_height :
                    # 检查特定节点，获取输入
                    lstm_input = None
                    lstm_input_type = None
                    if tree_table.rows[specific_node_index].left_pos is None:
                        lstm_input_type = 1
                        lstm_input = torch.tensor([1,0,0],dtype=torch.float).to(self.device).unsqueeze(0)
                    elif tree_table.rows[specific_node_index].right_pos is None:
                        lstm_input_type = 2
                        lstm_input = torch.tensor([0, 1, 0], dtype=torch.float).to(self.device).unsqueeze(0)
                    elif tree_table.rows[specific_node_index].father_pos is None:
                        lstm_input_type = 3
                        lstm_input = torch.tensor([0, 0, 1], dtype=torch.float).to(self.device).unsqueeze(0)
                    #print("输入：" + str(lstm_input)) 4
                    # 得到lstm的输出
                    output, (h_out, c_out) = self.lstm(lstm_input,(h_out,c_out))
                    # 符号概率的输出
                    output_symbol = self.fc_output(self.gaussian_activation(output))
                    # 得到符号的概率输出
                    output_symbol = F.softmax(output_symbol, dim=1)

                    # 得到终止节点的下标区域
                    star_index , end_index = getTerimIndex(self.symbol_set)
                    # 建立分布，采样
                    temp_dist = Categorical(output_symbol[0][star_index:])
                    action = temp_dist.sample()
                    action = action + torch.tensor(star_index)
                    dist = Categorical(output_symbol)
                    #action = dist.probs[star_index:].sample()
                    log_prob = dist.log_prob(action)
                    log_prob_list.append(log_prob)
                    #termi = output_symbol[0][star_index:]
                    # 获取本次生成的符号,使用argmax函数找到具有最大值的索引
                    #predicted_index = torch.argmax(termi).item() + star_index
                    # 在symbol_set中找到符号，填表
                    temp_symbol = self.symbol_set.symbol_list[action]
                    #print("输出：" + temp_symbol.name) 5
                    # 创建树表中的一行数据row,填序号，符号，左右节点和节点类型
                    new_row = Row(tree_table.length)
                    new_row.symbol = temp_symbol
                    new_row.arg_num = temp_symbol.arg_num
                    if new_row.arg_num == 0:  # 终止节点，左右都填-1
                        new_row.left_pos = -1
                        new_row.right_pos = -1
                    # 根据输入填特定节点的树表,即当前节点new_row是特定节点的什么节点
                    if lstm_input_type == 1:
                        tree_table.rows[specific_node_index].left_pos = new_row.position
                    elif lstm_input_type == 2:
                        tree_table.rows[specific_node_index].right_pos = new_row.position
                    elif lstm_input_type == 3:
                        tree_table.rows[specific_node_index].father_pos = -5
                        tree_table.rows[specific_node_index].root_type = True
                    # 根据输入填当前节点的树表
                    if lstm_input_type == 1 or lstm_input_type == 2:
                        new_row.father_pos = specific_node_index
                    elif lstm_input_type == 3:
                    # 已经不能向上生长,当前节点报废处理
                        new_row.left_pos = -1
                        new_row.right_pos = -1
                        new_row.father_pos = -1
                    tree_table.addRow(new_row)
                    # print("当前表：") 6
                    # tree_table.display()

                else:
                # 当树还没达到最大高度时，一切正常生长
                    # 检查特定节点，获取输入 TODO 获取一个Batch的输入，形状为1000*3 one-hot dtype = torch.int64
                    # tree-table , specific_node_index
                    lstm_input , lstm_input_type = getInput(tree_table,specific_node_index,self.batch_size)
                    print("正常生长 ： ")
                    print("lstm_input.shape = "+str(lstm_input.shape))
                    input()
                    #print("输入：" + str(lstm_input)) 7
                    # 得到lstm的输出
                    output, (h_out, c_out) = self.lstm(lstm_input,(h_out,c_out))
                    # 符号概率的输出
                    output_symbol = self.fc_output(self.gaussian_activation(output))
                    # 得到符号的概率输出
                    output_symbol = F.softmax(output_symbol, dim=1)
                    dist = Categorical(output_symbol)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    log_prob_list.append(log_prob)
                    # 获取本次生成的符号,使用argmax函数找到具有最大值的索引
                    #predicted_index = torch.argmax(output_symbol)
                    # 在symbol_set中找到符号，填表
                    temp_symbol = self.symbol_set.symbol_list[action]
                    #print("输出：" + temp_symbol.name) 8
                    # 创建树表中的一行数据row,填序号，符号，左右节点和节点类型
                    new_row = Row(tree_table.length)
                    new_row.symbol = temp_symbol
                    new_row.arg_num = temp_symbol.arg_num
                    if new_row.arg_num == 0:  # 终止节点，左右都填-1
                        new_row.left_pos = -1
                        new_row.right_pos = -1
                    elif new_row.arg_num == 1:  # 单操作符节点，右孩子填-1
                        new_row.right_pos = -1
                    # 根据输入填特定节点的树表,即当前节点new_row是特定节点的什么节点
                    if lstm_input_type == 1 :
                        tree_table.rows[specific_node_index].left_pos = new_row.position
                    elif lstm_input_type == 2 :
                        tree_table.rows[specific_node_index].right_pos = new_row.position
                    elif lstm_input_type == 3 :
                        tree_table.rows[specific_node_index].father_pos = new_row.position
                    #根据输入填当前节点的树表
                    if lstm_input_type == 1 or lstm_input_type == 2 :
                        new_row.father_pos = specific_node_index
                    elif lstm_input_type == 3 :
                        if new_row.symbol.arg_num == 0: # 如果是终止节点，直接报废，特定节点为根节点
                            new_row.right_pos = -1
                            new_row.left_pos = -1
                            new_row.father_pos = -1
                            tree_table.rows[specific_node_index].father_pos = -5
                            tree_table.rows[specific_node_index].root_type = True
                        elif new_row.symbol.arg_num == 1: # 如果是单操作节点，特定节点为当前节点左节点
                            new_row.left_pos = specific_node_index
                        else :
                            # 此时要计算位置概率来决定
                            # 位置概率的输出
                            output_pos = self.fc_position(self.gaussian_activation(output))
                            # 得到位置的概率输出
                            output_pos = F.softmax(output_pos, dim=1)
                            # 获取本次生成的位置,使用argmax函数找到具有最大值的索引
                            predicted_index = torch.argmax(output_pos)
                            #print("位置概率："+str(predicted_index)) 9
                            # 100，则特定节点是当前节点的左孩子
                            if predicted_index == 0 :
                                new_row.left_pos = specific_node_index
                            # 010，则特定节点是当前节点的右孩子
                            elif predicted_index == 1:
                                new_row.right_pos = specific_node_index
                            # 001，则特定节点是根节点，修改特定节点的father_pos为-5，root_type为True
                            elif predicted_index == 2:
                                tree_table.rows[specific_node_index].father_pos = -5
                                tree_table.rows[specific_node_index].root_type = True
                                # 接下来把当前节点进行报废处理，即左右父全填-1
                                new_row.right_pos = -1
                                new_row.left_pos = -1
                                new_row.father_pos = -1
                    tree_table.addRow(new_row)
                    # print("当前表：") 10
                    # tree_table.display()
                # 更新特定节点
                specific_node_index = getSpecificNode(tree_table,specific_node_index)
            #input()
            # 每次生长完一个节点后都计算一下树的高度
            tree_table.updateHeight()
            #print("当前树高："+str(tree_table.height))
            #tree_table.display()

        return tree_table , torch.stack(log_prob_list)

'''
