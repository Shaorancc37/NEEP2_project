import torch

from train import train


if __name__ == '__main__':


    train(name="Sphere5",Epoch=500,learning_rate = 1e-3,batch_size = 5,layer_num = 4,max_height=5)