import copy
import math
import sys
from queue import Queue

functionSet = ["+","-","*","/","sin","cos","e","ln"]
terminalSet = ["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12"]
singleSet = ["sin","cos","e","ln"]
adfSet = ["ADF1","ADF2","ADF3","ADF4","ADF5","ADF6","ADF7","ADF8","ADF9"]
adfInputSet = ["a","b"]

def judgeType(x):
    for item in singleSet:
        if x == item :
            return 3
    for item in functionSet:
        if x == item :
            return 2
    for item in adfSet :
        if x == item :
            return 2
    for item in adfInputSet:
        if x == item:
            return 1
    for item in terminalSet:
        if x == item :
            return 1
    return 0

# 节点类，将String转换为表达式树用到的节点Node
class node:
    val = ""
    leftNode = None
    rightNode = None
    type = -1  # 1为终止变量集，2为函数集中需要两个参数的函数 +，-，*，/，ADF符号等，3为函数集中需要1个参数的函数sin,cos等
    def __init__(self,value=None,left=None,right=None):
        self.val = value
        self.leftNode = left
        self.rightNode = right



def preMainRoot(main_root, root_adf):
    index = -1
    # 先判断当前main_root节点是否是ADF节点
    for item in adfSet:
        if main_root.val == item:
            # 是ADF节点的话，标记ADF下标
            index = int(item[-1])
    # 插入后可能是另一个ADF替换当前ADF，需要循环判断
    while index != -1:
        # 被标记过的话，调用insertADF将对应ADF子结构插入当前节点
        main_root = insertADF(main_root,copy.deepcopy(root_adf[index-1]))
        index = -1
        for item in adfSet:
            if main_root.val == item:
                # 是ADF节点的话，标记ADF下标
                index = int(item[-1])
    # 插入后，若有左右节点则继续遍历替换操作
    if main_root.leftNode != None:
        main_root.leftNode = preMainRoot(main_root.leftNode,copy.deepcopy(root_adf))
    if main_root.rightNode != None:
        main_root.rightNode = preMainRoot(main_root.rightNode,copy.deepcopy(root_adf))
    # 返回ADF子结构全部插入的树 的根节点
    return main_root

def insertADF(root,root_adf):
    a = copy.deepcopy(root.leftNode)
    b = copy.deepcopy(root.rightNode)
    root_adf = preAdf(root_adf,a,b)
    return root_adf

def preAdf(root_adf,a,b):

    if root_adf == None:
        return None
    #print(root_adf.val)
    if root_adf.val == "a":
        return copy.deepcopy(a)
    if root_adf.val == "b":
        return copy.deepcopy(b)
    root_adf.leftNode = preAdf(root_adf.leftNode,a,b)
    root_adf.rightNode = preAdf(root_adf.rightNode,a,b)
    return root_adf

def levelorder(root):
    q = Queue()
    q.put(root)
    print("层序遍历如下")
    while q.empty() is False:
        temp = q.get()
        print(temp.val+" type="+str(temp.type))
        if temp != None:
            if temp.leftNode != None:
                q.put(temp.leftNode)
            if temp.rightNode != None:
                q.put(temp.rightNode)



# 解码器，将表达式字符串转换为表达式树，输入：表达式串 ，头部长度，尾部长度           返回：表达式树的根节点
class decode:
    def __init__(self,express_string,len):
        self.express_string = express_string
        self.len = len

    def display(self,root):
        levelorder(root)

# 1 是终止 ， 2 是二元操作 ， 3是一元操作
    def star(self):
        index = 0
        root = node()
        root.val = self.express_string[index]
        if judgeType(self.express_string[index])==1:
            root.type = 1
        elif judgeType(self.express_string[index])==2:
            root.type = 2
        else :
            root.type = 3
        index = index + 1
        que = Queue()
        que.put(root)
        while que.empty()==False:
            temp = que.get()
            if temp.type == 3:
                if index == self.len:
                    break
                temp.leftNode = node()
                char = self.express_string[index]
                temp.leftNode.type = judgeType(char)
                temp.leftNode.val = char
                que.put(temp.leftNode)
                index +=1
            elif temp.type == 2:
                if index == self.len:
                    break
                temp.leftNode = node()
                char = self.express_string[index]
                temp.leftNode.type = judgeType(char)
                temp.leftNode.val = char
                que.put(temp.leftNode)
                index +=1
                if index == self.len:
                    break
                temp.rightNode = node()
                char = self.express_string[index]
                temp.rightNode.type = judgeType(char)
                temp.rightNode.val = char
                que.put(temp.rightNode)
                index +=1
        return root

    def reductionToADF(self,main_root,root_adf):
        temp = preMainRoot(main_root,root_adf)
        return temp

    def calculateFitness(self,root,x_value):
        try:
            ans = calculate(root,x_value)
            return ans
        except:
            levelorder(root)
            re = calculate(root,x_value)
            print("测试计算结果：")
            print(re)
            print(self.express_string)

# 计算树结构的类
def calculate(root,x_value):
    #try:
    # type=1是终端变量集，返回输入变量
    if root.type == 1:
        index = int(root.val[1:])
        return x_value[index-1]
    # type=2是2元操作符，计算得到左右节点值var1,var2，通过操作符计算出结果并返回上一层
    elif root.type == 2 :
        var1 = calculate(root.leftNode,x_value)
        var2 = calculate(root.rightNode,x_value)
        if root.val == "+":
            return add(var1,var2)
        elif root.val == "-":
            return sub(var1,var2)
        elif root.val == "*":
            return mul(var1,var2)
        elif root.val == "/":
            return div(var1,(var2))
    # type=3是1元操作符，计算得到左右节点值var1，通过操作符计算出结果并返回上一层
    elif root.type == 3:
        var1 = calculate(root.leftNode, x_value)
        if root.val == "sin":
            if math.isinf(var1) is True:
                return 1
            else:
                return math.sin(var1)
        elif root.val == "cos":
            if math.isinf(var1) is True:
                return 1
            else:
                return math.cos(var1)
        elif root.val == "e":
            if var1>12:
                return 1
            else :
                return math.exp(var1)
        elif root.val == "ln":
             try:
                 if var1<=0:
                     return 1
                 else:
                    return math.log(abs(var1), math.e)
             except:
                return 1

def add(data1, data2):
    return data1 + data2
def sub(data1, data2):
    return data1 - data2
def mul(data1, data2):
    return data1 * data2
def div(data1, data2):
    if data2 == 0.0:
        return 1
    else:
        return data1 / (data2 + sys.float_info.min)



if __name__ == '__main__':
    main = ["/","sin","/","sin","ADF1","-","-","-","x3","sin","x2","x4","x3","x5","x2","x5","x2","x1","x5"]
    adf = []
    adf.append(["*","ln","+","b","b","a","b"])
    adf.append(["e", "b", "-", "b", "b","a","b" ])
    # adf.append(["a", "b", "a", "a", "a", ])
    # adf.append(["a", "b", "a", "a", "a", ])
    # adf.append(["a", "b", "a", "a", "a", ])
    main_decoder = decode(main,len(main))
    adf_decoder_list = []
    for i in range(2):
        adf_decoder_list.append(decode(adf[i],len(adf[i])))
    main_root = main_decoder.star()
    adf_root = []
    for i in range(2):
        adf_root.append(adf_decoder_list[i].star())

    root = main_decoder.reductionToADF(main_root,adf_root)
    levelorder(root)
