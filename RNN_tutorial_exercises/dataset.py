import gym
import neurogym as ngym
import json
import tasks
import numpy as np

task_dict = {name: getattr(tasks, name) for name in dir(tasks) if isinstance(getattr(tasks, name), type)}

class myDataset():
    def __init__(self, env: ngym.TrialEnv, batch_size):
        self.env = env
        self.batch_size = batch_size

    def __call__(self):
        input = []
        target = []
        masks = []
        for i in range(self.batch_size):
            self.env.new_trial()
            ob, gt = self.env.ob, self.env.gt
            response_start_idx = self.env.start_t['go'] // self.env.dt
            grace_period = 100 // self.env.dt
            mask = np.zeros_like(gt)
            mask[:response_start_idx, 1:] = 1
            mask[:response_start_idx, 0] = 2
            mask[response_start_idx:response_start_idx+grace_period, 1:] = 0
            mask[response_start_idx+grace_period:, 1:] = 5
            mask[response_start_idx+grace_period:, 0] = 10
            input.append(ob)
            target.append(gt)
            masks.append(mask)
        max_len = max([len(x) for x in input])
        input = [np.pad(x, ((0, max_len - len(x)), (0, 0)), 'constant', constant_values=0) for x in input]
        target = [np.pad(x, ((0, max_len - len(x)), (0, 0)), 'constant', constant_values=0) for x in target]
        masks = [np.pad(x, ((0, max_len - len(x)), (0, 0)), 'constant', constant_values=0) for x in masks]
        input = np.array(input).transpose(1, 0, 2)
        target = np.array(target).transpose(1, 0, 2)
        masks = np.array(masks).transpose(1, 0, 2)
        return input, target, masks

def get_dataset(task_name, batch_size, **keywords) -> ngym.Dataset:
    if task_name in task_dict:
        env = task_dict[task_name]()
        dataset = myDataset(env, batch_size=batch_size)
        return dataset
    dataset = ngym.Dataset(task_name, env_kwargs=keywords, batch_size=batch_size)
    return dataset

class Dataset():
    def __init__(self):
        pass

    def load_from_file(self, file_path):
        self.trainsets = []
        self.evalsets = []
        self.alias = []
        dataconfigs = json.load(open(file_path, 'r'))
        for config in dataconfigs:
            self.trainsets.append(get_dataset(config['task_name'], config['batch_size'], **config['keywords']))
            self.evalsets.append(get_dataset(config['task_name'], config['batch_size'], **config['keywords']))
            self.alias.append(config['alias'])
        self.input_size = self.trainsets[0].env.observation_space.shape[0]
        self.output_size = self.trainsets[0].env.action_space.shape[0]
        for i in range(len(self.trainsets)):
            if self.trainsets[i].env.observation_space.shape[0] != self.input_size:
                raise ValueError(f"Input size of dataset {i} does not match the first dataset.")
            if self.trainsets[i].env.action_space.shape[0] != self.output_size:
                raise ValueError(f"Output size of dataset {i} does not match the first dataset.")

    def __getitem__(self, index):
        return self.trainsets[index], self.evalsets[index], self.alias[index]
    
    def __str__(self) -> str:
        info = f"Dataset list with {len(self.trainsets)} items:\n"
        for i in range(len(self.trainsets)):
            info += f"  {i}: {self.alias[i]}\n"
            info += f"    input size: {self.trainsets[i].env.observation_space.shape[0]}\n"
            info += f"    output size: {self.trainsets[i].env.action_space.shape[0]}\n"
            info += f"    batch size: {self.trainsets[i].batch_size}\n"
        return info