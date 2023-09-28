import torch
import torch.nn as nn

class W_block(nn.Module):
    def __init__(self,stride,in_dim,out_dim,num_points,seq_length):
        super().__init__()
        self.nodeWC1 = nn.Conv2d(in_channels=in_dim,
                                                   out_channels=out_dim,
                                                   kernel_size=(1,3),
                                                   padding=(0,1),
                                                   stride=(1,stride))
        self.nodeWC2 = nn.Conv2d(in_channels=in_dim,
                                                   out_channels=out_dim,
                                                   kernel_size=(1,3),
                                                   padding=(0,1),
                                                   stride=(1,stride))
        self.layer_norm = torch.nn.LayerNorm([out_dim, num_points, seq_length])

    def forward(self,x):
        '''
        input:b,c,n,t
        output:
        '''
        #逐点卷积
        sig = self.nodeWC1(x)
        tan = self.nodeWC2(x)
        sig = torch.sigmoid(sig)
        tan = torch.tanh(tan)
        x = sig*tan
        return self.layer_norm(x)

class W_Module(nn.Module):
    def __init__(self,num_points,seq_length):
        super(W_Module,self).__init__()
        self.dims = [
            [
                [1,4]
            ],
            [
                [4,4],
                # [4,4]
            ],
            [
                [4,4],
                # [4,4]
            ],
            [
                [4,4],
                # [4,4]
            ]
        ]
        self.strides = [1,2,2,2]
        self.stages = nn.ModuleList()
        #定义各个阶段
        l = seq_length
        for i in range(len(self.dims)):
            for dim in self.dims[i]:
                self.stages.append(
                    W_block(self.strides[i],dim[0],dim[1],num_points,l)
                )
            l = round(l/2)
    def forward(self,x):
        '''
        input:x:b,t,n,c
        output:b,c_out,n,t/8
        '''
        need_concat = []
        out = x.permute(0,3,2,1)
        for stage in self.stages:
            out = stage(out)
            need_concat.append(out)
        return need_concat,out


if __name__ == "__main__":
    #当前模块进行测试
    b,t,n,c = 8,12,10,1
    x = torch.randn(b,t,n,c)
    # w_block = W_block(1,1,4,n,t)
    # x = w_block(x.permute(0,3,2,1))
    w_module = W_Module(n,t)
    cats,x = w_module(x)
    print(x.size())
