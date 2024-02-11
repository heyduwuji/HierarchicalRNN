import torch
from torch import nn

class RNNCore(nn.Module):
    def __init__(self, hidden_size, tau, dt, activation_func=torch.relu, **keywords):
        super().__init__()
        self.input_size = None
        self.hidden_size = hidden_size
        self.back_size = None
        self.activation_func = activation_func
        if dt:
            alpha = dt / tau
        else:
            alpha = 1 / tau
        self.alpha = alpha

        self.W_in = None
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_b = None

    def connect(self, input_size, back_size):
        self.input_size = input_size
        self.back_size = back_size
        self.W_in = nn.Linear(input_size, self.hidden_size)
        self.W_b = nn.Linear(back_size, self.hidden_size) if back_size > 0 else None

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)
    
    def forward(self, input, back, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input.shape)
        
        if back is not None:
            h_new = self.activation_func(self.W_in(input) + self.W_h(hidden) + self.W_b(back))
        else:
            h_new = self.activation_func(self.W_in(input) + self.W_h(hidden))
        h_new = (1 - self.alpha) * hidden + self.alpha * h_new
        return h_new
    
class SequentialHRNN(nn.Module):
    def __init__(self, layers: nn.ModuleList, input_size, output_size, **keywords):
        super().__init__()
        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size
        self.classifier = nn.Linear(layers[-1].hidden_size, output_size)
        self.connect()

    def connect(self):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].connect(self.input_size, self.layers[i+1].hidden_size)
            elif i == len(self.layers) - 1:
                self.layers[i].connect(self.layers[i-1].hidden_size, self.output_size)
            else:
                self.layers[i].connect(self.layers[i-1].hidden_size, self.layers[i+1].hidden_size)

    def init_hidden(self, input_shape):
        for i in range(len(self.layers)):
            self.activity[i].append(self.layers[i].init_hidden(input_shape))

    def recurrence(self, input):
        out = []
        for i in range(len(self.layers)):
            if i == 0:
                hidden = self.layers[i](input, self.activity[i+1][-1], self.activity[i][-1])
            elif i == len(self.layers) - 1:
                hidden = self.layers[i](self.activity[i-1][-1], None, self.activity[i][-1])
            else:
                hidden = self.layers[i](self.activity[i-1][-1], self.activity[i+1][-1], self.activity[i][-1])
            out.append(hidden)
        return out

    def forward(self, input):
        self.activity = [[] for _ in range(len(self.layers))]
        step = input.size(0)
        self.init_hidden(input.shape)
        for i in range(step):
            hidden = self.recurrence(input[i])
            self.activity = [old + [new] for old, new in zip(self.activity, hidden)]
        output = torch.stack(self.activity[-1][1:], dim=0)
        pred = self.classifier(output)
        return pred, output