import pandas as pd

from utils.timefeatures import time_features

# start_date = '2016-07-01 02:00:00'
# end_date = '2017-11-24 17:00:00'
# # freq = '5T'  # 每小时生成一个时间点
# #
# # date_range = pd.date_range(start=start_date, freq=freq,periods=12)
# # print(date_range[-1])
# date_range = pd.date_range(start=start_date,end=end_date, freq='H')
# data_stamp = time_features(date_range, freq='T')
# data_stamp = data_stamp.transpose(1, 0)
# print(data_stamp[478:488])

import torch.nn as nn

import torch,einops
x = torch.ones((4,8,3,4))
y = torch.sum(x,dim=-1)
print(x,y)
