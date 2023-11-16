import torch

from RL_NEEP2.treetable import TreeTable , Row
from RL_NEEP2.node import Node
# 通过这个函数获取 特定节点、父节点、兄弟节点、当前生成节点位置
def getNowPosition(tree_table):
    specific_node = None
    father_node  = None
    bro_node = None
    now_node_pos = None

    len = tree_table.length
    for i in range(len):
        # 左孩子为None ， 则生成左孩子
        if tree_table.rows[i].left_pos is None:
            specific_node = i
            father_node = i
            if tree_table.rows[i].right_pos is None or tree_table.rows[i].right_pos == -1:
                bro_node = 0
            else:
                bro_node = tree_table.rows[i].right_pos
            now_node_pos = 1
            break
        # 右孩子为None, 则生成右孩子
        elif tree_table.rows[i].right_pos is None:
            specific_node = i
            father_node = i
            bro_node = tree_table.rows[i].left_pos
            now_node_pos = 2
            break
    # 如果表里都填满了，那就说明当前要生成的节点是无效的，那么就结构信息为None
    if specific_node is None:
        now_node_pos = 3

    # now_node_pos 1为左孩子、2为右孩子、3为无效节点
    return specific_node , father_node , bro_node , now_node_pos


def getInput(tree_table,batch_size,input_size,symbol_set):
    input_list = torch.tensor([], dtype=torch.int64)
    father_node_list = []
    bro_node_list = []
    now_node_pos_list = []

    for i in range(batch_size):
        father_node = None
        bro_node = None
        input_net = None
        # 树表为空
        if tree_table[i].length == 0:
            input_list = torch.cat((input_list, torch.ones(3, dtype=torch.int64)), dim=0)
            father_node_list.append(-5)
            bro_node_list.append(0)
            now_node_pos_list.append(0)
        # 树表不为空,我需要根据树表判断当前生成节点的位置，以此得到他的父节点和兄弟节点信息
        else:
            specific_node , father_node , bro_node , now_node_pos = getNowPosition(tree_table[i])
            father_node_list.append(father_node)
            bro_node_list.append(bro_node)
            now_node_pos_list.append(now_node_pos)
            # 如果生成的是左节点
            if now_node_pos == 1:
                input_list = torch.cat((input_list, torch.tensor([tree_table[i].rows[father_node].symbol_pos,tree_table[i].rows[bro_node].symbol_pos,now_node_pos], dtype=torch.int64).unsqueeze(0)), dim=0)
                #将父节点、兄弟节点连接起来作为输入
            elif now_node_pos == 2:
                input_list = torch.cat((input_list, torch.tensor([tree_table[i].rows[father_node].symbol_pos,tree_table[i].rows[bro_node].symbol_pos, now_node_pos], dtype=torch.int64).unsqueeze(0)),dim=0)
                # 将父节点、兄弟节点连接起来作为输入
            elif now_node_pos == 3:
                input_list = torch.cat((input_list, torch.tensor([0, 0, 3], dtype=torch.int64).unsqueeze(0)),dim=0)
                #将父节点、兄弟节点连接起来作为输入，都为0





    return input_list.reshape(batch_size,-1) , father_node_list , now_node_pos_list # 还有specific_node 和  now_node_pos的列表


def updateTreeTable(tree_table,father_list,now_nodepos_list,action,symbol_set,batch_size):

    for i in range(batch_size):
        # 表里无数据 ， 即当前要填的为根节点
        if tree_table[i].length == 0:
            # 在symbol_set中找到符号，填表
            temp_symbol = symbol_set.symbol_list[action[i]]
            # 创建树表中的一行数据row,填序号，符号，左右节点和节点类型
            new_row = Row(tree_table[i].length)
            new_row.symbol = temp_symbol
            new_row.arg_num = temp_symbol.arg_num
            new_row.symbol_pos = action[i]
            new_row.root_type = True
            new_row.father_pos = father_list[i]
            if new_row.arg_num == 0:  # 终止节点
                new_row.left_pos = -1
                new_row.right_pos = -1
            elif new_row.arg_num == 1:  # 单操作符节点
                new_row.right_pos = -1
            tree_table[i].addRow(new_row)
        # 表里有数据，则直接填表
        else:
            # 无效节点直接填
            if now_nodepos_list[i] == 3:
                # 在symbol_set中找到符号，填表
                temp_symbol = symbol_set.symbol_list[action[i]]
                # 创建树表中的一行数据row,填序号，符号，左右节点和节点类型
                new_row = Row(tree_table[i].length)
                new_row.symbol = temp_symbol
                new_row.arg_num = temp_symbol.arg_num
                new_row.symbol_pos = action[i]
                # 无效节点左右父节点皆为-1
                new_row.father_pos = -1
                new_row.left_pos = -1
                new_row.right_pos = -1
                tree_table[i].addRow(new_row)
            # 如果是有效节点
            # 先填父节点表，再填当前节点表
            else:
                # 在symbol_set中找到符号，填表
                temp_symbol = symbol_set.symbol_list[action[i]]
                # 创建树表中的一行数据row,填序号，符号，左右节点和节点类型
                new_row = Row(tree_table[i].length)
                new_row.symbol = temp_symbol
                new_row.arg_num = temp_symbol.arg_num
                new_row.symbol_pos = action[i]
                new_row.father_pos = father_list[i]
                if new_row.arg_num == 0:  # 终止节点
                    new_row.left_pos = -1
                    new_row.right_pos = -1
                elif new_row.arg_num == 1:  # 单操作符节点
                    new_row.right_pos = -1
                # TODO 根据 father_node 和 now_node_pos填父节点表
                # 如果是左节点
                if now_nodepos_list[i] == 1:
                    tree_table[i].rows[father_list[i]].left_pos = new_row.position
                # 如果是右节点
                elif now_nodepos_list[i] == 2:
                    tree_table[i].rows[father_list[i]].right_pos = new_row.position
                # TODO  根据 father_node 和 now_node_pos填当前节点表
                new_row.father_pos = father_list[i]
                tree_table[i].addRow(new_row)

    return tree_table
