from HRNN import RNNCore, SequentialHRNN
import torch
from dataset import Dataset, get_dataset
from trainer import Trainer
import json

def train(model_config_file, data_config_file, save_path, checkpoint=None):
    dataset = Dataset()
    dataset.load_from_file(data_config_file)
    model_config = json.load(open(model_config_file, 'r'))
    print(dataset)

    input_size = dataset.input_size
    output_size = dataset.output_size
    modules = []
    for config in model_config:
        modules.append(RNNCore(**config))
    net = SequentialHRNN(modules, input_size, output_size)
    if checkpoint:
        net.load_state_dict(torch.load(checkpoint))

    trainer = Trainer(net, dataset, Omega=None)
    n_iter = 50000
    lr = 0.01
    record_freq = 100
    net, Omega = trainer.train_all(n_iter, lr, record_freq, save_path+'_model', save_path+'_omega')
    import random
    print(random.choice(list(Omega.items())))
    # torch.save(net.state_dict(), save_path)
    print('trainning finished')

if __name__ == '__main__':
    train('model_config.json', 'data_config.json', './checkpoint/compare')