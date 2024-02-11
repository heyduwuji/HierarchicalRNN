import numpy as np
import torch

from CTRNN import RNNNet
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
dataset = get_dataset('PerceptualDecisionMaking-v0', seq_len, batch_size, **keywords)

# configure a RNN net to fit this dataset
input_size = dataset.env.observation_space.shape[0]
output_size = dataset.env.action_space.n
hidden_size = 128
tau = 40
print(f'model & dataset configuration: input_size={input_size}, output_size={output_size}, hidden_size={hidden_size},\n batch_size={batch_size}, tau={tau}, dt={dt}, stimulus={stimulus}, seq_len={seq_len}')
net = RNNNet(input_size, hidden_size, output_size, tau, dt)

# configure a trainner
trainer = Trainer(net, dataset)
n_iter = 1000
lr = 0.01
record_freq = 100
print(f'trainning configuration: n_iter={n_iter}, lr={lr}, record_freq={record_freq}')

# start trainning
net = trainer.train(n_iter, lr, record_freq)

# test and get neural activity
activity, trial_infos = trainer.test(200)
n_trial, seq_len, hidden_size = activity.shape
print(activity.shape)
activity = activity.reshape(-1, hidden_size)
pca = PCA(n_components=2)
pca.fit(activity)
activity_pc = pca.transform(activity)
activity_pc = activity_pc.reshape(n_trial, seq_len, -1)

# classify activity according to predicted action
act = [[], [], []]
for i in range(n_trial):
    act[trial_infos[i]['pred']].append(activity_pc[i, :, :])
print(f'number of trials for each action: {len(act[0])}, {len(act[1])}, {len(act[2])}')

# plot one line per trial for each action
fig, ax = plt.subplots(1, 3)
for i in range(3):
    for j in range(len(act[i])):
        ax[i].plot(act[i][j][:, 0], act[i][j][:, 1], 'o-', alpha=0.1)
        ax[i].plot(act[i][j][-1, 0], act[i][j][-1, 1], '^', alpha=0.5)
    ax[i].set_title(f'action {i}')
plt.savefig('./fig1/PDM3.png')
