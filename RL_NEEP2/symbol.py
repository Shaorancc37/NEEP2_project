import numpy as np

class Symbol:

    def __init__(self, function, name, arg_num, x_index=None):
        self.function = function  # 符号对应的函数初始化
        self.name = name  # 符号的名称初始化
        self.arg_num = arg_num  # 符号操作变量的数量初始化
        self.x_index = x_index  # 符号的位置

    def __call__(self, *args):
        return self.function(*args)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

class SymbolLibrary:
    def __init__(self,symbol_list):
        # 符号集列表：函数符号 + 终止符号
        self.symbol_list = symbol_list
        #  符号集的名称列表
        self.names = [t.name for t in symbol_list]
        # 符号集数量
        self.length = len(symbol_list)
        # 创建一个包含符号操作数数量的列表
        self.arg_nums = np.array([t.arg_num for t in symbol_list], dtype=np.int32)
        # 创建一个列表，存输入变量在symbol_list中的位置下标
        self.input_symbols = np.array([i for i, t in enumerate(self.symbol_list) if t.x_index is not None])



def protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)

def protected_exp(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 < 100, np.exp(x1), 0.0)

def protected_log(x1):
    """Closure of log for non-positive arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)