import torch
import torch.nn as nn

class GraphOperation(nn.Module):
    def __init__(self,num_points,in_dim):
        super().__init__()
        self.GCNs = nn.ModuleList()
        for i in range(num_points):
            self.GCNs.append(nn.Linear(in_dim, in_dim, bias=True))
    def forward(self,R):
        '''
        input:
            R:b,c4,n,n
        output:b,c4,n,1
        '''
        b,c4,n = R.size(0),R.size(1),R.size(2)
        mx,_ = torch.max(R,dim=1)
        A = torch.relu(torch.tanh(mx))#b,n,n
        # 进行图卷积操作
        Fg = torch.zeros(b,c4,n)
        for k in range(n):
            x = torch.einsum('bcn, bn->bc', R[:,:,:,k], A[:,k,:])
            Fg[:,:,k] = self.GCNs[k](x)
        Fg = Fg.unsqueeze(-1)
        return Fg

def RelationalFeatureExtraction(S,F4):
        '''
        input:
            S:b,n,n,l4
            F4:b,c4,n,l4
        output:b,c4,n,n
        '''
        b,n,l4 = S.size(0),S.size(1),S.size(3)
        c4 = F4.size(1)
        Se = S.unsqueeze(2).expand(b, n, c4, n, l4)#b,n,c4,n,t
        R = torch.zeros(b,c4,n,n)
        for k in range(n):
            rk = 0
            for j in range(l4):
                rk+=Se[:,k,:,:,j]*F4[:,:,:,j]
            R[:,:,:,k] = rk
        return R

class SpatioTemporalCorrelationComputation(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//4,kernel_size=(1,1))
    def forward(self,x):
        '''
        input:b,c4,n,l4
        output:b,n,n,l4
        '''
        Fc = self.conv(x)#b,c4/4,n,l4
        Fl = Fc[:,:,:,-1]#b,c4/4,n
        #计算Fl和Fc的余弦相似性
        xy = torch.einsum('bcn, bcmt->bnmt', Fl, Fc)
        xx = torch.einsum('bcn, bcn->bc', Fl, Fl).sqrt()
        yy = torch.einsum('bcnt, bcnt->bc', Fc, Fc).sqrt()
        l2 = torch.einsum('bc, bc->b', xx, yy).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        res = torch.div(xy, l2)
        return res

class ES_Module(nn.Module):
    def __init__(self,num_points,in_dim):
        super().__init__()
        self.STCC = SpatioTemporalCorrelationComputation(in_dim)
        self.GCN = GraphOperation(num_points,in_dim)

    def forward(self,x):
        '''
        input:(F4) b,c4,n,l4
        output:b,c4,n,1
        '''
        S = self.STCC(x)
        R = RelationalFeatureExtraction(S,x)
        x = self.GCN(R)
        return x

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

class ESGCN(nn.Module):
    def __init__(self,num_points,hidden,seq_length,out_dim):
        super(ESGCN, self).__init__()
        self.w_module = W_Module(num_points,seq_length)
        self.es_module = ES_Module(num_points,hidden)
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
    model = ESGCN(n,dim,t,c)
    x = model(x)
    print(x.size())
