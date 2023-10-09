import functorch.dim
import torch
import torch.nn as nn
import torch.nn.functional as F
from neep2.treetable import TreeTable , Row
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


def getTerimIndex(symbol_set):
    temp = symbol_set.input_symbols
    star = temp[0]
    end = temp[-1]
    return star , end


class ActorNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, output_size, symbol_set , max_height, device):
        super(ActorNet, self).__init__()
        # 定义LSTM网络结构
        self.lstm = nn.LSTM(input_size,hidden_size, num_layers, batch_first=True)
        # 定义输出层
            # fc_output是符号对应的输出
        self.fc_output = nn.Linear(hidden_size,output_size)
            # fc_position是当生成父节点时，判断特定节点对应父节点位置的输出
        self.fc_position = nn.Linear(hidden_size,3)
        # 高斯激活函数
        self.gaussian_activation = GaussianActivation()
        self.symbol_set = symbol_set
        self.max_height = max_height
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self):

        # 初始化lstm的隐藏层输出 和 细胞状态输出
        h_out = torch.rand(self.num_layers, self.num_layers, self.hidden_size).to(self.device)
        c_out = torch.rand(self.num_layers, self.num_layers, self.hidden_size).to(self.device)
        # 初始化树表
        tree_table = TreeTable()
        # 初始化state列表
        state_list = []
        # 初始化action列表
        action_list = []
        # 初始化log_p分布
        log_prob_list = []
        # 初始化特定节点
        specific_node_index = None
        # 开始构造树
        while tree_table.judge() and tree_table.height<=self.max_height+1:
            if tree_table.height == 0:
                lstm_input = torch.tensor([0,0,0],dtype=torch.float).to(self.device).unsqueeze(0)
                state_list.append(lstm_input)
                #print("输入："+str(lstm_input)) 1
                output,(h_out,c_out) = self.lstm(lstm_input)
                # 符号概率的输出
                output_symbol = self.fc_output(self.gaussian_activation(output))
                # 得到符号的概率输出
                output_symbol = F.softmax(output_symbol,dim=1)
                dist = Categorical(output_symbol)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                log_prob_list.append(log_prob)
                # print("dist : "+str(dist.probs))
                # print("action : "+str(action))
                # print("log_prob : "+str(log_prob))
                # input()

                action_list.append(output_symbol)
                # 获取本次生成的符号,使用argmax函数找到具有最大值的索引
                predicted_index = torch.argmax(output_symbol)
                # 在symbol_set中找到符号，填表
                temp_symbol = self.symbol_set.symbol_list[action]
                #print("输出："+temp_symbol.name) 2
                # 创建树表中的一行数据row,填序号，符号，左右节点和节点类型
                new_row = Row(tree_table.length)
                new_row.symbol = temp_symbol
                new_row.arg_num = temp_symbol.arg_num
                if new_row.arg_num ==0 : #终止节点
                    new_row.left_pos = -1
                    new_row.right_pos = -1
                elif new_row.arg_num == 1 : # 单操作符节点
                    new_row.right_pos = -1
                tree_table.addRow(new_row)
                # print("当前表：") 3
                # tree_table.display()
                # 更新特定节点
                specific_node_index = 0
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
                    state_list.append(lstm_input)
                    #print("输入：" + str(lstm_input)) 4
                    # 得到lstm的输出
                    output, (h_out, c_out) = self.lstm(lstm_input,(h_out,c_out))
                    # 符号概率的输出
                    output_symbol = self.fc_output(self.gaussian_activation(output))
                    # 得到符号的概率输出
                    output_symbol = F.softmax(output_symbol, dim=1)

                    action_list.append(output_symbol)
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
                    # 检查特定节点，获取输入
                    lstm_input = None
                    lstm_input_type = None
                    if tree_table.rows[specific_node_index].left_pos is None:
                        lstm_input_type = 1
                        lstm_input = torch.tensor([1, 0, 0], dtype=torch.float).to(self.device).unsqueeze(0)
                    elif tree_table.rows[specific_node_index].right_pos is None:
                        lstm_input_type = 2
                        lstm_input = torch.tensor([0, 1, 0], dtype=torch.float).to(self.device).unsqueeze(0)
                    elif tree_table.rows[specific_node_index].father_pos is None:
                        lstm_input_type = 3
                        lstm_input = torch.tensor([0, 0, 1], dtype=torch.float).to(self.device).unsqueeze(0)
                    state_list.append(lstm_input)
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
                    action_list.append(output_symbol)
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


