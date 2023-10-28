import torch
from torch import nn

class CTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, tau=100, dt=None, **keywords):
        super.__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if dt:
            alpha = dt / tau
        else:
            alpha = 1 / tau
        self.alpha = alpha

        ## bias or not?
        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        h_new = torch.relu(self.input2hidden(input) + self.hidden2hidden(hidden))
        h_new = (1 - self.alpha) * hidden + self.alpha * h_new
        return h_new

    def forward(self, input, hidden=None):
        # input have size (seq_len, batch_size, embedding_size)
        if hidden is None:
            hidden = self.init_hidden(input.shape)

        output = []
        step = input.size(0)
        for i in step:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, tau=100, dt=None, **keywords):
        super.__init__()
        self.rnn = CTRNN(input_size, hidden_size, tau, dt, **keywords)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, _ = self.rnn(input)
        pred = self.fc(output)
        return pred, output