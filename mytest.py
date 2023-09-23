import math

import torch
import time
import numpy as np
import os
import einops
def CostMatrix(x):
    '''
    Cost Matrix Calculation
    input:N,T
    output:N,N
    '''
    CMC = torch.zeros((x.size(0) + 1, x.size(0) + 1))
    for i in range(1, x.size(0) + 1):
        arr = torch.zeros(CMC.size(0)-i)
        for j in range(i, x.size(0) + 1):
            xy = torch.sum(torch.mul(x[i - 1], x[j - 1]),dim=0)
            xx = torch.sum(torch.mul(x[i - 1], x[i - 1]),dim=0)
            yy = torch.sum(torch.mul(x[j - 1], x[j - 1]),dim=0)
            dpq = torch.div(xy, torch.mul(torch.sqrt(xx), torch.sqrt(yy)))
            # CMC[i][j] = CMC[j][i] = dpq + min(CMC[i - 1][j - 1], CMC[i - 1][j], CMC[i][j - 1])
            arr[j-i] = dpq + min(CMC[i - 1][j - 1], CMC[i - 1][j], CMC[i][j - 1])
        max_values = torch.max(arr)
        torch.div(arr, max_values)
        for j in range(i, x.size(0) + 1):
            CMC[i][j] = CMC[j][i] = arr[j-i]
    CMC = CMC[1:, 1:]
    return CMC
def A_temporal(x):
    #转为tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x).to(device)
    x = einops.rearrange(x, "T N C->N (T C)")
    At = torch.zeros((x.size(0),x.size(0)))
    CMC = CostMatrix(x)
    for i in range(x.size(0)):
            for j in range(i,x.size(0)):
                At[i][j] = At[j][i] = CMC[i][j]
    #保存到目录
    return At
    # torch.save(At,'./at.pt')
x = torch.randn((12,6,1))#T N C
# print(x)
# mean = x.mean()
# std = x.std()
# print((x-mean)/std)

A_temporal(x)

# x = torch.load('D:\program\python\STSGCN_Pytorch\data\processed\PEMS04-1\A_temporal.pt')


