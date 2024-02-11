import numpy as np
import torch

from TwoLayerRNN import LayerRNNNet
from dataset import get_dataset
from trainer import Trainer

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# configure a PDM dataset
task_name = 'PerceptualDecisionMaking-v0'
seq_len = 100
batch_size = 16
dt = 20
stimulus = 1000
keywords = {'dt': dt, 'timing': {'stimulus': stimulus}}
# keywords = {'dt': dt}
dataset = get_dataset('PerceptualDecisionMaking-v0', seq_len, batch_size, **keywords)

# configure a RNN net to fit this dataset
input_size = dataset.env.observation_space.shape[0]
output_size = dataset.env.action_space.n
hidden_size = [128, 64]
tau = torch.tensor([40, 60])
print(f'model & dataset configuration: input_size={input_size}, output_size={output_size}, hidden_size={hidden_size},\n batch_size={batch_size}, tau={tau}, dt={dt}, stimulus={dataset.env.timing}, seq_len={seq_len}')
net = LayerRNNNet(input_size, hidden_size, output_size, tau, dt)

# configure a trainner
trainer = Trainer(net, dataset)
n_iter = 5000
lr = 0.01
record_freq = 100
print(f'trainning configuration: n_iter={n_iter}, lr={lr}, record_freq={record_freq}')

# start trainning
net = trainer.train(n_iter, lr, record_freq)
torch.save(net.state_dict(), './checkpoint/twolayer_re.pth')
print('trainning finished')
