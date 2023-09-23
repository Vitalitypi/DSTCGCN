import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.reset_gate = nn.Linear(hidden_size*2, hidden_size)
        self.update_gate = nn.Linear(input_size*2, hidden_size)
        self.new_memory = nn.Linear(input_size*2, hidden_size)
        self.output_gate = nn.Linear(hidden_size, input_size)
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=2)
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        combined_new = torch.cat((input, reset * hidden), dim=2)
        new_memory = torch.tanh(self.new_memory(combined_new))#h'

        output = update * hidden + (1 - update) * new_memory
        yt = torch.sigmoid(self.output_gate(output))
        return output,yt

    def init_hidden(self, batch_size,num_size):
        return torch.zeros(batch_size,num_size, self.hidden_size)

rnn = GRU(64, 64)
input = torch.randn(32, 307, 64)
h0 = torch.randn(32, 307, 64)
output, hn = rnn(input, h0)
print(output.size(),hn.size())
# torch.zeros()
