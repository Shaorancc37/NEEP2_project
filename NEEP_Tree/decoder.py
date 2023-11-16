from queue import Queue

from RL_NEEP2.node import Node


def getString(index, action, symbol_set):
    string = []
    for item in action:
        string.append(symbol_set.symbol_list[item[index].item()])
    return string


def leve(root):
    q = Queue()
    q.put(root)
    print("层序遍历")
    while q.empty() is False:
        temp = q.get()
        print(temp.symbol)
        if temp != None:
            if temp.left_child!=None:
                q.put(temp.left_child)
            if temp.right_child!=None:
                q.put(temp.right_child)


class Decoder:
    def __init__(self,index,action,symbol_set):
        self.string = getString(index,action,symbol_set)
        self.len = len(action)
        self.symbol_set = symbol_set


    def getString(self):
        ans = []
        for item in self.string:
            ans.append(item.name)
        return ans


    def creat(self):
        index = 0
        root = Node(self.string[index])
        index = index +1
        que = Queue()
        que.put(root)
        while que.empty()==False:
            temp = que.get()
            if temp.symbol.arg_num == 1:
                if index == self.len:
                    break
                temp.left_child = Node(self.string[index])
                que.put(temp.left_child)
                index = index + 1
            elif temp.symbol.arg_num == 2:
                if index == self.len:
                    break
                temp.left_child = Node(self.string[index])
                que.put(temp.left_child)
                index = index + 1
                if index == self.len:
                    break
                temp.right_child = Node(self.string[index])
                que.put(temp.right_child)
                index = index+1
        #leve(root)
        return root




