import torch
import multiprocessing
from train import train
from functools import partial
import time

problemSet = {
        #"问题名称":["问题名称",数据集大小,输入变量数],
         "Sphere5":["Sphere5"],
         "Poly10": ["Poly10"],
        "Dic1":["Dic1"],
        "Dic3": ["Dic3"],
        "Dic4": ["Dic4"],
        "Dic5": ["Dic5"],
        # "Nico4":["Nico4"],
        # "Nico6": ["Nico6"],
        # "Nico9": ["Nico9"],
        # "Nico13": ["Nico13"],
        #"Nico14": ["Nico14"],
        # "Nico15": ["Nico15"],
        #"Nico16": ["Nico16"],
        # "Nico19": ["Nico19"],
        # "Nico20": ["Nico20"],
        # "Nico21": ["Nico21"],
        #"Koza1": ["Koza1"],
        # "Pagie1":["Pagie1"],
        #"Vlad3":["Vlad3"],
        # "Nguyen6":["Nguyen6"],
        # "Nguyen7":["Nguyen7"]
        # "Yacht":["Yacht"],
        # "Airfoil":["Airfoil"],
        # "Concrete":["Concrete"],
        # "Energy":["Energy"],
        # "FishToxicity":["FishToxicity"],
        #"Forestfires":["Forestfires"]
    }

if __name__ == '__main__':
    # 开始计时
    start_time = time.time()
    # 获取cpu数目
    nb_cpu = multiprocessing.cpu_count()
    # # 创建cpu进程池
    pool = multiprocessing.Pool(nb_cpu - 6)

    # 设置Epoch 、 独立重复实验次数
    batch_size = 50
    layer_num = 1
    learning_rate = 1e-3
    Epoch = 250  # 代数
    count = 30  # 独立重复次数
    cou = []
    for i in range(count):
        cou.append(i)
    # 对problemSet里的每个实验进行30次独立重复实验
    # 可以根据自身需要对实验进行删减，直接将probleSet里的实验进行注释即可
    for problem in problemSet:
        do = partial(train,name=problemSet[problem][0],Epoch=Epoch,learning_rate = learning_rate,batch_size=batch_size,layer_num=layer_num)
        # 并行计算
        pool.map(do, cou)
        # print(str(problem) + " 任务完成")
        # print("当前时间：" + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    # 结束计时
    end_time = time.time()
    # 计算时间差
    execution_time = end_time - start_time
    # 输出结果
    print(f"执行时间：{execution_time}秒")


    #train(name="Sphere5",Epoch=2,learning_rate = 1e-3,batch_size = 50,layer_num = 1,count=0)