from HRNN import RNNCore, SequentialHRNN
import torch
from dataset import get_dataset
from trainer import Trainer
import json

def train(config_file, save_path, checkpoint=None):
    model_config = json.load(open(config_file, 'r'))
    task_name = 'PerceptualDecisionMaking-v0'
    seq_len = 100
    batch_size = 16
    dt = 20
    stimulus = 1000
    keywords = {'dt': dt, 'timing': {'stimulus': stimulus}}
    dataset = get_dataset('PerceptualDecisionMaking-v0', seq_len, batch_size, **keywords)
    eval_set = get_dataset('PerceptualDecisionMaking-v0', seq_len, batch_size, **keywords)

    input_size = dataset.env.observation_space.shape[0]
    output_size = dataset.env.action_space.n
    modules = []
    for config in model_config:
        modules.append(RNNCore(**config))
    net = SequentialHRNN(modules, input_size, output_size)
    if checkpoint:
        net.load_state_dict(torch.load(checkpoint))

    trainer = Trainer(net, dataset, eval_set, Omega=None)
    n_iter = 5000
    lr = 0.01
    record_freq = 100
    net, Omega = trainer.train(n_iter, lr, record_freq)
    import random
    print(random.choice(list(Omega.items())))
    torch.save(net.state_dict(), save_path)
    print('trainning finished')

if __name__ == '__main__':
    train('model_config.json', './checkpoint/test.pth')