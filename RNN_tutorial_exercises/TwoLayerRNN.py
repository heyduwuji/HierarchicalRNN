import torch
from torch import nn

class TwoLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size: list, tau=torch.tensor([100, 100]), dt=None, **keywords):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if dt:
            alpha = dt / tau
        else:
            alpha = 1 / tau
        self.alpha = alpha


        self.input2hidden_1 = nn.Linear(input_size, hidden_size[0])
        self.hidden2hidden_1 = nn.Linear(hidden_size[0], hidden_size[0])

        self.input2hidden_2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden2hidden_2 = nn.Linear(hidden_size[1], hidden_size[1])

    def init_hidden(self, input_shape, nlayer):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size[nlayer])
    
    def recurrence(self, input, hidden_1, hidden_2):
        h1_new = torch.relu(self.input2hidden_1(input) + self.hidden2hidden_1(hidden_1))
        h1_new = (1 - self.alpha[0]) * hidden_1 + self.alpha[0] * h1_new

        h2_new = torch.relu(self.input2hidden_2(h1_new) + self.hidden2hidden_2(hidden_2))
        h2_new = (1 - self.alpha[1]) * hidden_2 + self.alpha[1] * h2_new

        return h1_new, h2_new
    
    def forward(self, input, hidden_1=None, hidden_2=None):
        if hidden_1 is None:
            hidden_1 = self.init_hidden(input.shape, 0)
        if hidden_2 is None:
            hidden_2 = self.init_hidden(input.shape, 1)

        output_1 = []
        output_2 = []
        step = input.size(0)
        for i in range(step):
            hidden_1, hidden_2 = self.recurrence(input[i], hidden_1, hidden_2)
            output_1.append(hidden_1)
            output_2.append(hidden_2)

        output_1 = torch.stack(output_1, dim=0)
        output_2 = torch.stack(output_2, dim=0)
        return output_1, output_2, hidden_1, hidden_2
    
class LayerRNNNet(nn.Module):
    def __init__(self, input_size, hidden_size:list, output_size, tau=[100, 100], dt=None, **keywords):
        super().__init__()
        self.rnn = TwoLayerRNN(input_size, hidden_size, tau, dt, **keywords)
        self.fc = nn.Linear(hidden_size[-1], output_size)
    
    def forward(self, input):
        output_1, output_2, hidden_1, hidden_2 = self.rnn(input)
        pred = self.fc(output_2)
        return pred, [output_1, output_2]