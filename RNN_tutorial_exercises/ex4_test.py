import numpy as np
import torch

from TwoLayerRNN import LayerRNNNet
from dataset import get_dataset
from trainer import Trainer
from torch import nn
import torch.optim as optim
import neurogym as ngym

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def find_fix(net: nn.Module):
    for param in net.parameters():
        param.requires_grad = False

    batch_size = 64

    input = np.tile([1, .5, .5], (batch_size, 1))
    input = torch.tensor(input, dtype=torch.float32)
    hidden_1 = torch.tensor(np.random.rand(batch_size, hidden_size_list[0])*3, requires_grad=True, dtype=torch.float32)
    hidden_2 = torch.tensor(np.random.rand(batch_size, hidden_size_list[1])*3, requires_grad=True, dtype=torch.float32)
    optimizer = optim.Adam([hidden_1], lr=0.001)
    criterion = nn.MSELoss()

    running_loss = 0
    for i in range(10000):
        optimizer.zero_grad()   # zero the gradient buffers
        
        # Take the one-step recurrent function from the trained network
        new_h_1, new_h_2 = net.rnn.recurrence(input, hidden_1, hidden_2)
        loss = criterion(new_h_1, hidden_1)
        loss.backward()
        optimizer.step()    # Does the update

        running_loss += loss.item()
        if i % 1000 == 999:
            running_loss /= 1000
            print('Step {}, Loss {:0.4f}'.format(i+1, running_loss))
            running_loss = 0
    
    fixedpoint = hidden_1.detach().numpy()
    print("fixedpoint shape: ", fixedpoint.shape)
    return fixedpoint

# configure a PDM dataset
task_name = 'PerceptualDecisionMaking-v0'
seq_len = 100
batch_size = 16
dt = 20
stimulus = 2000
keywords = {'dt': dt, 'timing': {'stimulus': stimulus}}
# keywords = {'dt': dt}
dataset = get_dataset('PerceptualDecisionMaking-v0', seq_len, batch_size, **keywords)

# configure a RNN net to fit this dataset
input_size = dataset.env.observation_space.shape[0]
output_size = dataset.env.action_space.n
hidden_size_list = [128, 64]
tau = torch.tensor([40, 60])
print(f'model & dataset configuration: input_size={input_size}, output_size={output_size}, hidden_size={hidden_size_list},\n batch_size={batch_size}, tau={tau}, dt={dt}, stimulus={dataset.env.timing}, seq_len={seq_len}')
net = LayerRNNNet(input_size, hidden_size_list, output_size, tau, dt)
net.load_state_dict(torch.load('./checkpoint/twolayer_re.pth'))
print(net.state_dict().keys())
plt.figure()
plt.imshow(net.state_dict()['rnn.input2hidden_1.weight'].detach().numpy())
plt.colorbar()
plt.savefig('./fig1/input2hidden_1.png')
plt.figure()
plt.imshow(net.state_dict()['rnn.hidden2hidden_1.weight'].detach().numpy())
plt.colorbar()
plt.savefig('./fig1/hidden2hidden_1.png')
plt.figure()
plt.imshow(net.state_dict()['rnn.input2hidden_2.weight'].detach().numpy())
plt.colorbar()
plt.savefig('./fig1/input2hidden_2.png')
plt.figure()
plt.imshow(net.state_dict()['rnn.hidden2hidden_2.weight'].detach().numpy())
plt.colorbar()
plt.savefig('./fig1/hidden2hidden_2.png')

# configure a trainner
trainer = Trainer(net, dataset)
n_iter = 1000
lr = 0.01
record_freq = 100
print(f'trainning configuration: n_iter={n_iter}, lr={lr}, record_freq={record_freq}')

# test and get neural activity
activity, trial_infos = trainer.test(200)
n_trial, seq_len, hidden_size = activity[1].shape
print(activity[1].shape)

# act_2 = activity[1].reshape(n_trial * seq_len, hidden_size)
# pca = PCA(n_components=2)
# pca.fit(act_2)
# plt.figure()
# for i in range(n_trial):
#     act_pc = pca.transform(activity[1][i])
#     trial = trial_infos[i]
#     color = 'red' if trial['ground_truth'] == 0 else 'blue'
#     plt.scatter(act_pc[0, 0], act_pc[0, 1], marker='+', color=color)
#     plt.plot(act_pc[:, 0], act_pc[:, 1], color=color, alpha=0.1)
#     plt.scatter(act_pc[-1, 0], act_pc[-1, 1], marker='*', color=color)
# fixedpoint = find_fix(net)
# fixedpoints_pc = pca.transform(fixedpoint)
# plt.plot(fixedpoints_pc[:, 0], fixedpoints_pc[:, 1], 'x')

# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.savefig('./fig1/hidden2.png')

# n_trial, seq_len, hidden_size = activity[1].shape
# print(activity[1].shape)

n_trial, seq_len, hidden_size = activity[0].shape
act_1 = activity[0].reshape(n_trial * seq_len, hidden_size)
pca = PCA(n_components=2)
pca.fit(act_1)
plt.figure()
for i in range(n_trial):
    act_pc = pca.transform(activity[0][i])
    trial = trial_infos[i]
    color = 'red' if trial['ground_truth'] == 0 else 'blue'
    plt.scatter(act_pc[0, 0], act_pc[0, 1], marker='+', color=color)
    plt.plot(act_pc[:, 0], act_pc[:, 1], color=color, alpha=0.1)
    plt.scatter(act_pc[-1, 0], act_pc[-1, 1], marker='*', color=color)
fixedpoint = find_fix(net)
fixedpoints_pc = pca.transform(fixedpoint)
plt.plot(fixedpoints_pc[:, 0], fixedpoints_pc[:, 1], 'x')

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.savefig('./fig1/hidden1.png')