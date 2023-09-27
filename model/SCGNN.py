import einops
import torch
import torch.nn as nn

class gcn_operation(nn.Module):
    def __init__(self, adj,temporal_adj, in_dim, out_dim, num_vertices, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param temporal_adj: 时间图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.adj = adj*temporal_adj
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {'GLU', 'relu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x):
        """
        :param x: (B,N,C)
        :param mask:(3*N, 3*N)
        :return: (B,N,C)
        """

        x = torch.einsum('nm, bmc->bnc', self.adj.to(x.device), x)  # (B,N,C)
        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # (B,N,C)
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # (B,N,C)

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out

        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # (B,N,C)

class CNN1_12(nn.Module):
    def __init__(self, channels,horizon,num_of_vertices):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels*num_of_vertices,
                                                   out_channels=3*channels*num_of_vertices,
                                                   kernel_size=(1,3),
                                                   stride=(1,3))
        self.conv2 = nn.Conv2d(in_channels=3*channels*num_of_vertices,
                                                   out_channels=6*channels*num_of_vertices,
                                                   kernel_size=(1,2),
                                                   stride=(1,2))
        self.conv3 = nn.Conv2d(in_channels=6*channels*num_of_vertices,
                                                   out_channels=12*channels*num_of_vertices,
                                                   kernel_size=(1,2),
                                                   stride=(1,2))
        self.FC_1 = nn.Linear(3*channels*num_of_vertices, 3*channels*num_of_vertices, bias=True)
        self.FC_2 = nn.Linear(6*channels*num_of_vertices, 6*channels*num_of_vertices, bias=True)
        self.horizon = horizon
        self.channels = channels
        self.num_of_vertices = num_of_vertices
    def forward(self,x):
        """
        :param x: B T N C
        :return: B, TinN, N, Cout
        """
        x = einops.rearrange(x, 'b (d1 d2) n c -> b d1 d2 (n c)', d1=1, d2=12)
        x = x.permute(0,3,1,2)#B C H W
        x = self.conv1(x)#B 3C H W
        x = torch.relu(self.FC_1(x.permute(0,2,3,1))).permute(0,3,1,2)
        x = self.conv2(x)
        x = torch.relu(self.FC_2(x.permute(0,2,3,1))).permute(0,3,1,2)
        x = self.conv3(x)#B 12C 1 1
        x = einops.rearrange(x, 'b (t n1 c1) n c -> b t (n n1) (c c1)', t=self.horizon, n1=self.num_of_vertices,c1=self.channels)
        return x

class CNN2_6(nn.Module):
    def __init__(self, channels,horizon,num_of_vertices):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels*num_of_vertices,
                                                   out_channels=3*channels*num_of_vertices,
                                                   kernel_size=(2,3),
                                                   stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=3*channels*num_of_vertices,
                                                   out_channels=6*channels*num_of_vertices,
                                                   kernel_size=(1,2),
                                                   stride=(1,2))
        self.conv3 = nn.Conv2d(in_channels=6*channels*num_of_vertices,
                                                   out_channels=12*channels*num_of_vertices,
                                                   kernel_size=(1,2),
                                                   stride=(1,2))
        self.FC_1 = nn.Linear(3*channels*num_of_vertices, 3*channels*num_of_vertices, bias=True)
        self.FC_2 = nn.Linear(6*channels*num_of_vertices, 6*channels*num_of_vertices, bias=True)
        self.horizon = horizon
        self.channels = channels
        self.num_of_vertices = num_of_vertices
    def forward(self,x):
        """
        :param x: B T N C
        :return: B, TinN, N, Cout
        """
        x = einops.rearrange(x, 'b (d1 d2) n c -> b d1 d2 (n c)', d1=2, d2=6)
        x = x.permute(0,3,1,2)#B C H W
        x = self.conv1(x)#B 3C H W
        x = torch.relu(self.FC_1(x.permute(0,2,3,1))).permute(0,3,1,2)
        x = self.conv2(x)
        x = torch.relu(self.FC_2(x.permute(0,2,3,1))).permute(0,3,1,2)
        x = self.conv3(x)#B 12C 1 1
        x = einops.rearrange(x, 'b (t n1 c1) n c -> b t (n n1) (c c1)', t=self.horizon, n1=self.num_of_vertices,c1=self.channels)
        return x

class CNN3_4(nn.Module):
    def __init__(self, channels,horizon,num_of_vertices):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels*num_of_vertices,
                                                   out_channels=3*channels*num_of_vertices,
                                                   kernel_size=(2,2),
                                                   stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=3*channels*num_of_vertices,
                                                   out_channels=6*channels*num_of_vertices,
                                                   kernel_size=(2,2),
                                                   stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=6*channels*num_of_vertices,
                                                   out_channels=12*channels*num_of_vertices,
                                                   kernel_size=(1,2),
                                                   stride=(1,2))
        self.FC_1 = nn.Linear(3*channels*num_of_vertices, 3*channels*num_of_vertices, bias=True)
        self.FC_2 = nn.Linear(6*channels*num_of_vertices, 6*channels*num_of_vertices, bias=True)
        self.horizon = horizon
        self.channels = channels
        self.num_of_vertices = num_of_vertices
    def forward(self,x):
        """
        :param x: B T N C
        :return: B, TinN, N, Cout
        """
        x = einops.rearrange(x, 'b (d1 d2) n c -> b d1 d2 (n c)', d1=3, d2=4)
        x = x.permute(0,3,1,2)#B C H W
        x = self.conv1(x)#B 3C H W
        x = torch.relu(self.FC_1(x.permute(0,2,3,1))).permute(0,3,1,2)
        x = self.conv2(x)
        x = torch.relu(self.FC_2(x.permute(0,2,3,1))).permute(0,3,1,2)
        x = self.conv3(x)#B 12C 1 1
        x = einops.rearrange(x, 'b (t n1 c1) n c -> b t (n n1) (c c1)', t=self.horizon, n1=self.num_of_vertices,c1=self.channels)
        return x


class SCGNN(nn.Module):
    def __init__(self, adj, temporal_adj, num_of_vertices, in_dim,out_dims, first_layer_embedding_size, activation='GLU',
                 horizon=12):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.out_dims = out_dims
        self.activation = activation
        self.First_FC = nn.Linear(self.in_dim, first_layer_embedding_size, bias=True)
        self.in_dim = first_layer_embedding_size
        self.Layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.Layers.append(CNN1_12(self.in_dim,horizon,num_of_vertices))
        self.gates.append(nn.Linear(self.in_dim, self.out_dim, bias=True))
        self.Layers.append(CNN2_6(self.in_dim,horizon,num_of_vertices))
        self.gates.append(nn.Linear(self.in_dim, self.out_dim, bias=True))
        self.Layers.append(CNN3_4(self.in_dim,horizon,num_of_vertices))
        self.gates.append(nn.Linear(self.in_dim, self.out_dim, bias=True))
        self.GCNs = nn.ModuleList()
        self.horizon = horizon
        for i in range(horizon):
            gcn_operations = nn.ModuleList()
            gcn_operations.append(
                gcn_operation(
                    adj=adj,
                    temporal_adj = temporal_adj,
                    in_dim=self.in_dim,
                    out_dim=self.out_dims[0],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )
            for i in range(1, len(self.out_dims)):
                gcn_operations.append(
                    gcn_operation(
                        adj=adj,
                        temporal_adj = temporal_adj,
                        in_dim=self.out_dims[i-1],
                        out_dim=self.out_dims[i],
                        num_vertices=self.num_of_vertices,
                        activation=self.activation
                    )
                )
            self.GCNs.append(gcn_operations)
            self.Second_FC = nn.Linear(first_layer_embedding_size, self.in_dim, bias=True)
            self.Third_FC = nn.Linear(self.in_dim, self.in_dim, bias=True)
    def forward(self,x):
        """
        :param x: B, Tin, N, Cin)
        :return: B, Tout, N, Cout
        """
        x = torch.relu(self.First_FC(x))  # B, Tin, N, Cout
        for i in range(self.horizon):
            x_ = x[:,i,:,:]
            for j in range(len(self.out_dims)):
                x_ = self.GCNs[i][j](x_)
            x[:,i,:,:] = x_
        out = torch.zeros(x.size())
        for i in range(3):
            gate = torch.tanh(self.gates[i](x))
            conv = self.Layers[i](x)
            out += (gate+1)*conv
        out = self.Third_FC(torch.relu(self.Second_FC(out)))
        out = einops.rearrange(out, 'b t n c -> b t (n c)')
        return out
if __name__ == "__main__":
    #x: B, Tin, N, Cin
    #将数据进行切分
    cin = 1
    dim = 4
    num_of_vertices = 307
    Tin = 12
    B = 32
    first_layer_embedding_size=4
    adj = torch.ones((num_of_vertices,num_of_vertices))
    x = torch.randn(B,Tin,num_of_vertices,cin)#T N B C
    scgnn = SCGNN(adj,adj,num_of_vertices,in_dim=1,out_dims=[4, 4, 4],first_layer_embedding_size=4)
    # cnn12 = CNN1_12(dim,12,num_of_vertices)
    # x = cnn12(x)
    x = scgnn(x)
    print(x,x.size())


# print(history, num_of_vertices, in_dim, hidden_dims,
#                  first_layer_embedding_size, out_layer_dim, activation, use_mask,
#                  temporal_emb, spatial_emb, horizon, strides)
# 4 307 64 [[64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64]] 64 128 GLU True True True 12 3
