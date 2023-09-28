import pandas as pd

from utils.timefeatures import time_features

import torch.nn as nn

import torch, einops

b, n, l4 = 2, 3, 1
c4 = 3
Se = torch.randn(b,n,c4,n,l4)
F4 = torch.randn(b,c4,n,l4)
R = torch.zeros(b,c4,n,n)
for k in range(n):
    rk = 0
    for j in range(l4):
        rk += Se[:, k, :, :, j] * F4[:, :, :, j]
    R[:, :, :, k] = rk
R_ = torch.einsum('bkcmj,bcmj->bcmk', Se, F4)
print(R, R_)
