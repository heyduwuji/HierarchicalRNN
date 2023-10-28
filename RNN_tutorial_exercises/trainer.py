import torch
from torch import nn
import torch.optim as optim
import neurogym as ngym
import time

class Trainer():
    def __init__(self, model: nn.Module, dataset: ngym.Dataset):
        self.model = model
        self.dataset = dataset

    def train(self, iter, lr):
        optimizer = optim.Adam(self.model.parameters(), lr)
        loss_func = nn.CrossEntropyLoss()
        running_loss = 0
        running_acc = 0
        self.dataset.env.reset()
        for i in range(iter):
            input, label = self.dataset()
            input = torch.from_numpy(input)