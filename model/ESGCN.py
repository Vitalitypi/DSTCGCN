import torch
import W_Module
import ES_Module
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,num_points,hidden,seq_length,out_dim):
        super(Model, self).__init__()
        self.w_module = W_Module.W_Module(num_points,seq_length)
        self.es_module = ES_Module.ES_Module(num_points,hidden)
        self.Cis = nn.ModuleList()
        for i in range(4):
            self.Cis.append(nn.Conv2d(in_channels=hidden,
                                                   out_channels=hidden,
                                                   kernel_size=(1,1)))
        self.Ce = nn.Conv2d(in_channels=hidden,
                                                   out_channels=hidden,
                                                   kernel_size=(1,1))
        self.FC1 = nn.Linear(hidden,hidden,True)
        self.FC2 = nn.Linear(hidden,out_dim,True)

    def forward(self,x):
        '''
        input:x b,t,n,c
        '''
        cats,F4 = self.w_module(x) # b,c4,n,l4
        Fg = self.es_module(F4)
        out = self.Ce(Fg)
        for i in range(4):
            res = self.Cis[4-i-1](cats[4-i-1])
            if 4-i-1!=0:
                out = torch.cat((out,res),dim=-1)
            else:
                out += res
        Y = self.FC2(torch.relu(self.FC1(out.permute(0,3,2,1))))
        return Y

if __name__ == "__main__":
    #当前模块进行测试
    b,t,n,c = 32,12,307,1
    dim = 4
    x = torch.randn(b,t,n,c)
    model = Model(n,dim,t,c)
    x = model(x)
    print(x.size())
