from RL_NEEP_ALL.node import Node


# 根据树表tree_table来递归建立树
def creatTree(tree_table, root_index):
    #print(root_index)
    # print("table_length = "+str(tree_table.length))
    # print("root_index =  "+str(root_index))
    node = Node(tree_table.rows[root_index].symbol)
    if node.symbol.arg_num == 0:
        return node
    if tree_table.rows[root_index].left_pos is not None and tree_table.rows[root_index].left_pos >= 0 :
        node.left_child = creatTree(tree_table,tree_table.rows[root_index].left_pos)
    if tree_table.rows[root_index].right_pos is not None and tree_table.rows[root_index].right_pos >= 0 :
        node.right_child = creatTree(tree_table,tree_table.rows[root_index].right_pos)
    return node

# 计算当前树的高度
def getHeight(root):
    # 如果根节点为空，高度为0
    if root is None:
        return 0

    # 递归计算左子树和右子树的高度
    left_height = getHeight(root.left_child)
    right_height = getHeight(root.right_child)

    # 返回左右子树中较大的高度，并加上根节点自身高度（1）
    return max(left_height, right_height) + 1


class TreeTable:

    def __init__(self):
        self.rows = []
        self.length = 0
        self.height = 0

    def addRow(self,row):
        self.rows.append(row)
        self.length = self.length + 1

    def display(self):
        for item in self.rows:
            print(str(item.position)+"  "+str(item.left_pos)+" "+str(item.right_pos)+" "+str(item.father_pos)+" "+str(item.symbol.name)+" "+str(item.arg_num)+" "+str(item.root_type))

    def judge(self):
        if self.length == 0 :
            return True
        elif self.length>0 :
            for item in self.rows:
                if item.left_pos is None :
                    return True
                if item.right_pos is None :
                    return True
                if item.father_pos is None :
                    return True
            return False

    def updateHeight(self):
        # 找到当前树表中的根节点root_index
        root_index = -1
        # 先遍历树表看是否有已确定的根节点
        for i in range(self.length):
            if self.rows[i].root_type is True:
                root_index = i
                break
        # 若无确立的根，则找到当前的根
        if root_index == -1 :
            root_index = 0
            for item in self.rows:
                if item.father_pos == None:
                    break
                root_index = root_index + 1
        # 以当前根节点来建立树
        root = creatTree(self,root_index)
        # 计算当前树的高度,这个高度就是当前最大高度
        self.height = getHeight(root)

    def decodeTreeTable(self):
        # 找到当前树表中的根节点root_index
        root_index = -1
        # 先遍历树表看是否有已确定的根节点
        for i in range(self.length):
            if self.rows[i].root_type is True:
                root_index = i
                break
        # 以当前根节点来建立树
        root = creatTree(self, root_index)
        return root

    def getSolution(self):
        ans = []
        for item in self.rows:
            ans.append(item.symbol.name)
        return ans


class Row:

    def __init__(self,position,left_pos=None,right_pos=None,father_pos=None,symbol=None,symbol_pos=None,arg_num=0,root_type=False):
        self.position = position
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.father_pos = father_pos
        self.symbol = symbol
        self.symbol_pos = symbol_pos
        self.arg_num = arg_num
        self.root_type = root_type
