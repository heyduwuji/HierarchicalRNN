import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask):
        # target has shape (seq_len, batch_size, output_size)
        loss = input - target
        loss = loss * mask
        loss = loss ** 2
        loss = torch.mean(loss)
        return loss