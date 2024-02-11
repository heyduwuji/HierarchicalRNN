import torch
from torch import nn
import torch.optim as optim
import neurogym as ngym
import time
import numpy as np

class Trainer():
    def __init__(self, model: nn.Module, dataset: ngym.Dataset, eval_set: ngym.Dataset, Omega: dict[str, torch.Tensor]):
        self.model = model
        self.dataset = dataset
        self.eval_set = eval_set
        self.output_size = dataset.env.action_space.n
        self.loss_func = nn.CrossEntropyLoss()
        self.Omega = Omega if Omega is not None else {k: torch.zeros_like(v) for k, v in self.model.named_parameters() if v.requires_grad}

    def train(self, iter, lr, record_freq, eval_freq=1000, c=1.0, ksi=0.01):
        optimizer = optim.Adam(self.model.parameters(), lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        running_loss = 0
        running_acc = 0
        self.dataset.env.reset()
        start_time = time.time()
        # 遍历self.model的所有可训练参数并保存到字典中用于后续检索历史参数
        self.last_param = {k: v.clone() for k, v in self.model.named_parameters() if v.requires_grad}
        model_params = {k: v.clone() for k, v in self.model.named_parameters() if v.requires_grad}
        omega = {k: torch.zeros_like(v) for k, v in self.model.named_parameters() if v.requires_grad}
        delta = {k: torch.zeros_like(v) for k, v in self.model.named_parameters() if v.requires_grad}
        
        for i in range(iter):
            input, label = self.dataset()
            # input have shape (seq_len, batch_size, input_size)
            # label have shape (seq_len, batch_size)
            input = torch.from_numpy(input)
            label = torch.from_numpy(label.flatten())

            optimizer.zero_grad()
            output, _ = self.model(input)
            # output have shape (seq_len, batch_size, output_size)
            output = output.view(-1, self.output_size)
            loss = self.loss_func(output, label)
            if c is not None:
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        loss += c * torch.sum(self.Omega[k] * (v - self.last_param[k]) ** 2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if c is not None and c > 0:
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        omega[k] = omega[k] + (v.grad * (v - model_params[k]))
                        delta[k] = delta[k] + (v - model_params[k])
                        model_params[k] = v.clone()

            # record running loss, print every 100 iters
            # But WHY running loss?
            running_loss += loss.item()
            if i % record_freq == record_freq - 1:
                running_loss /= record_freq
                print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                    i+1, running_loss, time.time() - start_time))
                running_loss = 0
            if i % eval_freq == eval_freq - 1:
                self.eval(100)

        self.Omega = {k: v + omega[k] / (delta[k] ** 2 + ksi) for k, v in self.Omega.items()}
        return self.model, self.Omega
    
    def eval(self, n_trail):
        env = self.eval_set.env
        env.reset(no_step=True)
        eval_loss = 0
        self.model.eval()
        total = 0
        correct = 0
        for i in range(n_trail):
            input, label = self.eval_set()
            input = torch.from_numpy(input)
            label = torch.from_numpy(label.flatten())
            output, _ = self.model(input)
            output = output.view(-1, self.output_size)
            loss = self.loss_func(output, label)
            eval_loss += loss.item()
            _, pred = torch.max(output, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
        eval_loss /= n_trail
        print('Eval Loss {:0.4f}, Eval Acc {:0.4f}'.format(eval_loss, correct / total))

    def test(self, n_trail):
        env = self.dataset.env
        env.reset(no_step=True)
        activity = {}
        trial_infos = []

        # run one trial per time, namely one batch per time
        for i in range(n_trail):
            trial_info = env.new_trial()
            ob, gt = env.ob, env.gt
            input = torch.from_numpy(ob[:, np.newaxis, :])
            pred, act = self.model(input)
            # pred have shape (seq_len, 1, output_size)
            # act have shape (seq_len, 1, hidden_size)
            pred = pred.detach().numpy()[:, 0, :]
            choice = np.argmax(pred[-1, :])
            correct = choice == gt[-1]
            for k in range(len(act)):
                if k not in activity:
                    activity[k] = []
                activity[k].append(act[k][:, 0, :].detach().numpy())
            trial_info.update({'pred': choice, 'correct': correct, 'seq_len': ob.shape[0]})
            trial_infos.append(trial_info)
        
        # print sample trials
        for i in range(5):
            print('Trial ', i, trial_infos[i])

        print('Average performance', np.mean([x['correct'] for x in trial_infos]))

        # activity have shape (n_trial, seq_len, hidden_size)
        return [np.stack(activity[k], axis=0) for k in range(len(activity))], trial_infos