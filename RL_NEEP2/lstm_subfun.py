import warnings

import torch
import numpy as np
from torch.distributions import Categorical
from treetable import TreeTable , Row
warnings.filterwarnings("ignore", category=UserWarning)

def getTerimIndex(symbol_set):
    temp = symbol_set.input_symbols
    star = temp[0]
    end = temp[-1]
    return star , end

def getInput(tree_table,specific_node_index,batch_size,One_Hot_length):
    #print(specific_node_index)
    input_list = torch.tensor([],dtype=torch.int64)
    input_type_list = []
    for i in range(batch_size):
        if tree_table[i].length == 0 :
            father = torch.zeros(One_Hot_length,dtype=torch.int64)
            bro = torch.zeros(One_Hot_length,dtype=torch.int64)
            input_list = torch.cat((input_list,torch.ones(3, dtype=torch.int64),father,bro),dim=0)
        else :
            if tree_table[i].rows[specific_node_index[i]].left_pos is None:

                input_type_list.append(1)
                father = tree_table[i].rows[specific_node_index[i]].symbol_pos
                bro = torch.zeros(One_Hot_length,dtype=torch.int64)
                father = torch.nn.functional.one_hot(torch.tensor(father), One_Hot_length)
                temp = torch.cat((father,bro,torch.tensor([1, 0, 0], dtype=torch.int64)),dim=0).unsqueeze(0)
                input_list = torch.cat((input_list,temp),dim=0)
            elif tree_table[i].rows[specific_node_index[i]].right_pos is None:

                input_type_list.append(2)
                father = tree_table[i].rows[specific_node_index[i]].symbol_pos
                bro = tree_table[i].rows[specific_node_index[i]].left_pos
                bro = tree_table[i].rows[bro].symbol_pos
                father = torch.nn.functional.one_hot(torch.tensor(father), One_Hot_length)
                bro = torch.nn.functional.one_hot(torch.tensor(bro), One_Hot_length)

                temp = torch.cat((father, bro, torch.tensor([0, 1, 0], dtype=torch.int64)), dim=0).unsqueeze(0)
                input_list = torch.cat((input_list, temp), dim=0)
            elif tree_table[i].rows[specific_node_index[i]].father_pos is None:

                input_type_list.append(3)
                left = tree_table[i].rows[specific_node_index[i]].left_pos
                right = tree_table[i].rows[specific_node_index[i]].right_pos

                if left == -1:
                    left = torch.zeros(One_Hot_length,dtype=torch.int64)
                else:
                    left = tree_table[i].rows[left].symbol_pos
                    left = torch.nn.functional.one_hot(torch.tensor(left), One_Hot_length)
                if right == -1:
                    right = torch.zeros(One_Hot_length,dtype=torch.int64)
                else:
                    right = tree_table[i].rows[right].symbol_pos
                    right = torch.nn.functional.one_hot(torch.tensor(right), One_Hot_length)

                temp = torch.cat((left, right, torch.tensor([0, 0, 1], dtype=torch.int64)), dim=0).unsqueeze(0)
                input_list = torch.cat((input_list, temp), dim=0)
            else:
                input_type_list.append(4)
                father = torch.zeros(One_Hot_length, dtype=torch.int64)
                bro = torch.zeros(One_Hot_length, dtype=torch.int64)
                temp = torch.cat((father, bro, torch.tensor([0, 1, 0], dtype=torch.int64)), dim=0).unsqueeze(0)
                input_list = torch.cat((input_list, temp), dim=0)


    return input_list.reshape(batch_size,-1) , input_type_list

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
    return specific_node_index



def updateTreeTable(lstm_input_type ,symbol_set,output_symbol,action1,action2,tree_table,specific_node_index,batch_size,max_height,finished):

    for i in range(batch_size):
        # 如果该树已经完成生成，则直接跳过
        if finished[i] == 1:
            continue
        else:  # 否则 按照规则来填写树表
            if tree_table[i].length == 0:
                #在symbol_set中找到符号，填表
                temp_symbol = symbol_set.symbol_list[action1[i]]
                # 创建树表中的一行数据row,填序号，符号，左右节点和节点类型
                new_row = Row(tree_table[i].length)
                new_row.symbol = temp_symbol
                new_row.arg_num = temp_symbol.arg_num
                new_row.symbol_pos = action1[i]
                if new_row.arg_num == 0:  # 终止节点
                    new_row.left_pos = -1
                    new_row.right_pos = -1
                elif new_row.arg_num == 1:  # 单操作符节点
                    new_row.right_pos = -1
                tree_table[i].addRow(new_row)
                specific_node_index[i] = 0
            # 当前树高度已经达到max_height ，向上无法生长节点，直接判定为根节点，向下只能生长终止节点
            elif tree_table[i].height >= max_height :
                #   重新进行采样，只能取终止节点
                # 得到终止节点的下标区域
                star_index, end_index = getTerimIndex(symbol_set)
                # 建立分布，采样
                temp_dist = Categorical(output_symbol[i][star_index:])
                action = temp_dist.sample()
                action = action + torch.tensor(star_index)
                # 在action1中更新action
                action1[i] = action
                # 在symbol_set中找到符号，填表
                temp_symbol = symbol_set.symbol_list[action1[i]]
                # 创建树表中的一行数据row,填序号，符号，左右节点和节点类型
                new_row = Row(tree_table[i].length)
                new_row.symbol = temp_symbol
                new_row.arg_num = temp_symbol.arg_num
                new_row.symbol_pos = action1[i]
                if new_row.arg_num == 0:  # 终止节点，左右都填-1
                    new_row.left_pos = -1
                    new_row.right_pos = -1
                # 根据输入填特定节点的树表,即当前节点new_row是特定节点的什么节点
                if lstm_input_type[i] == 1:
                    tree_table[i].rows[specific_node_index[i]].left_pos = new_row.position
                elif lstm_input_type[i] == 2:
                    tree_table[i].rows[specific_node_index[i]].right_pos = new_row.position
                elif lstm_input_type[i] == 3:
                    tree_table[i].rows[specific_node_index[i]].father_pos = -5
                    tree_table[i].rows[specific_node_index[i]].root_type = True
                # 根据输入填当前节点的树表
                if lstm_input_type[i] == 1 or lstm_input_type[i] == 2:
                    new_row.father_pos = specific_node_index[i]
                elif lstm_input_type[i] == 3:
                    # 已经不能向上生长,当前节点报废处理
                    new_row.left_pos = -1
                    new_row.right_pos = -1
                    new_row.father_pos = -1
                tree_table[i].addRow(new_row)
            else : # 当树还没达到最大高度时，一切正常生长
                # 在symbol_set中找到符号，填表
                temp_symbol = symbol_set.symbol_list[action1[i]]
                # 创建树表中的一行数据row,填序号，符号，左右节点和节点类型
                new_row = Row(tree_table[i].length)
                new_row.symbol = temp_symbol
                new_row.arg_num = temp_symbol.arg_num
                new_row.symbol_pos = action1[i]
                if new_row.arg_num == 0:  # 终止节点，左右都填-1
                    new_row.left_pos = -1
                    new_row.right_pos = -1
                elif new_row.arg_num == 1:  # 单操作符节点，右孩子填-1
                    new_row.right_pos = -1
                # 根据输入填特定节点的树表,即当前节点new_row是特定节点的什么节点
                if lstm_input_type[i] == 1:
                    tree_table[i].rows[specific_node_index[i]].left_pos = new_row.position
                elif lstm_input_type[i] == 2:
                    tree_table[i].rows[specific_node_index[i]].right_pos = new_row.position
                elif lstm_input_type[i] == 3:
                    tree_table[i].rows[specific_node_index[i]].father_pos = new_row.position
                # 根据输入填当前节点的树表
                if lstm_input_type[i] == 1 or lstm_input_type[i] == 2:
                    new_row.father_pos = specific_node_index[i]
                elif lstm_input_type[i] == 3:
                    if new_row.symbol.arg_num == 0:  # 如果是终止节点，直接报废，特定节点为根节点
                        new_row.right_pos = -1
                        new_row.left_pos = -1
                        new_row.father_pos = -1
                        tree_table[i].rows[specific_node_index[i]].father_pos = -5
                        tree_table[i].rows[specific_node_index[i]].root_type = True
                    elif new_row.symbol.arg_num == 1:  # 如果是单操作节点，特定节点为当前节点左节点
                        new_row.left_pos = specific_node_index[i]
                    else:
                        # 100，则特定节点是当前节点的左孩子
                        if action2[i] == 0:
                            new_row.left_pos = specific_node_index[i]
                        # 010，则特定节点是当前节点的右孩子
                        elif action2[i] == 1:
                            new_row.right_pos = specific_node_index[i]
                        # 001，则特定节点是根节点，修改特定节点的father_pos为-5，root_type为True
                        elif action2[i] == 2:
                            tree_table[i].rows[specific_node_index[i]].father_pos = -5
                            tree_table[i].rows[specific_node_index[i]].root_type = True
                            # 接下来把当前节点进行报废处理，即左右父全填-1
                            new_row.right_pos = -1
                            new_row.left_pos = -1
                            new_row.father_pos = -1
                tree_table[i].addRow(new_row)
        # 填完之后更新特定节点
        specific_node_index[i] = getSpecificNode(tree_table[i],specific_node_index[i])


    return tree_table,specific_node_index,action1

def updateFinished(tree_table,finished,batch_size):
    for i in range(batch_size):
        temp_flag = False
        for j in range(tree_table[i].length):
            if tree_table[i].rows[j].left_pos is None:
                temp_flag = True
            elif tree_table[i].rows[j].right_pos is None:
                temp_flag = True
            elif tree_table[i].rows[j].father_pos is None:
                temp_flag = True

        if temp_flag is False:
            finished[i] = 1
    return finished