import torch
import torch.nn as nn


class GraphOperation(nn.Module):
    def __init__(self,num_points,in_dim):
        super().__init__()
        self.GCN = nn.Linear(in_dim, in_dim, bias=True)
        # for i in range(num_points):
        #     self.GCNs.append(nn.Linear(in_dim, in_dim, bias=True))
    def forward(self,R):
        '''
        input:
            R:b,c4,n,n
        output:b,c4,n,1
        '''
        mx,_ = torch.max(R,dim=1)
        A = torch.relu(torch.tanh(mx))#b,n,n
        # 进行图卷积操作
        x = torch.einsum('bcnk, bkn->bck', R, A)
        Fg = self.GCN(x.permute(0,2,1)).permute(0,2,1)
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
        R_ = torch.einsum('bkcmj,bcmj->bcmk', Se, F4)
        print(R,R_)
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

if __name__ == "__main__":
    #当前模块进行测试
    b,c,n,t = 8,4,10,2
    x = torch.randn(b,c,n,t)
    es = ES_Module(n,c)
    x = es(x)
    # w_module = W_Module(n,t)
    # cats,x = w_module(x)
    print(x.size())
