import torch
from torch import nn
import torch.optim as optim
import neurogym as ngym
import time
import numpy as np
import dataset
from lossfunc import MaskedMSELoss

class Trainer():
    def __init__(self, model: nn.Module, datasets: dataset.Dataset, Omega: dict[str, torch.Tensor]):
        self.model = model
        self.datasets = datasets
        self.output_size = datasets.output_size
        self.loss_func = MaskedMSELoss()
        self.Omega = Omega if Omega is not None else {k: torch.zeros_like(v) for k, v in self.model.named_parameters() if v.requires_grad}

    def train(self, train_set, eval_set, iter, lr, record_freq, ckp_dir, omega_dir, eval_freq=1000, c=1.0, ksi=0.01):
        optimizer = optim.Adam(self.model.parameters(), lr)
        running_loss = 0
        running_acc = 0
        train_set.env.reset()
        start_time = time.time()
        # 遍历self.model的所有可训练参数并保存到字典中用于后续检索历史参数
        self.last_param = {k: v.clone() for k, v in self.model.named_parameters() if v.requires_grad}
        model_params = {k: v.clone() for k, v in self.model.named_parameters() if v.requires_grad}
        omega = {k: torch.zeros_like(v) for k, v in self.model.named_parameters() if v.requires_grad}
        delta = {k: torch.zeros_like(v) for k, v in self.model.named_parameters() if v.requires_grad}
        
        best_score = 0
        epoch = 0
        best_epoch = 0

        for i in range(iter):
            input, target, mask, _ = train_set()
            # input have shape (seq_len, batch_size, input_size)
            # label have shape (seq_len, batch_size, output_size)
            input = torch.from_numpy(input)
            target = torch.from_numpy(target)
            mask = torch.from_numpy(mask)

            optimizer.zero_grad()
            output, _ = self.model(input)
            # output have shape (seq_len, batch_size, output_size)
            loss = self.loss_func(output, target, mask)
            if c is not None:
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        loss += c * torch.sum(self.Omega[k].detach() * (v - self.last_param[k].detach()) ** 2)
            loss.backward()
            optimizer.step()
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    omega[k] = omega[k] + (-v.grad * (v - model_params[k]))
                    delta[k] = delta[k] + (v - model_params[k])
                    model_params[k] = v.clone()

            # record running loss, print every 100 iters
            running_loss += loss.item()
            if i % record_freq == record_freq - 1:
                running_loss /= record_freq
                print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                    i+1, running_loss, time.time() - start_time))
                running_loss = 0
            if i % eval_freq == eval_freq - 1:
                score = self.eval(eval_set, 100)
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), ckp_dir+f'best_{epoch}.pth')
                    torch.save(self.Omega, omega_dir+f'best_{epoch}.pth')
                print('Best score {:0.4f} at epoch {}'.format(best_score, best_epoch))
                epoch += 1

        self.Omega = {k: v + omega[k] / (delta[k] ** 2 + ksi) for k, v in self.Omega.items()}
        return self.model, self.Omega
    
    def popvec(self, y):
        """Population vector read out.

        Assuming the last dimension is the dimension to be collapsed

        Args:
            y: population output on a ring network. Numpy array (Batch, Units)

        Returns:
            Readout locations: Numpy array (Batch,)
        """
        pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
        temp_sum = y.sum(axis=-1)
        temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
        temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
        loc = np.arctan2(temp_sin, temp_cos)
        return np.mod(loc, 2*np.pi)
    
    def get_perf(self, output, angle, target):
        output_loc = output[-1, :, 1:]
        output_fix = output[-1, :, 0]
        output_loc = self.popvec(output_loc)
        # target_loc = [target[last[i], i, 1:] for i in range(len(last))]
        # target_loc = np.stack(target_loc, axis=0)
        # target_loc = self.popvec(target_loc)
        # target_dist = np.abs(target_loc - angle)
        # dist = np.minimum(target_dist, 2*np.pi - target_dist)
        # correct_loc = dist < 0.2 * np.pi
        # print(correct_loc)
        # target_fix = [target[last[i], i, 0] for i in range(len(last))]
        # target_fix = np.stack(target_fix, axis=0)
        # print(target.shape, target_fix.shape)
        # print(target_fix > 0.5)

        fixating = output_fix > 0.5
        
        original_dist = np.abs(output_loc - angle)
        dist = np.minimum(original_dist, 2*np.pi - original_dist)
        correct_loc = dist < 0.2 * np.pi

        perf = correct_loc * (1 - fixating)
        return perf.mean()
    
    def eval(self, eval_set, n_trail):
        env = eval_set.env
        env.reset(no_step=True)
        eval_loss = 0
        perf = 0
        self.model.eval()
        for i in range(n_trail):
            input, label, mask, angle = eval_set()
            input = torch.from_numpy(input)
            label = torch.from_numpy(label)
            mask = torch.from_numpy(mask)
            output, _ = self.model(input)
            loss = self.loss_func(output, label, mask)
            eval_loss += loss.item()
            perf += self.get_perf(output.detach().numpy(), angle, label.detach().numpy())
        eval_loss /= n_trail
        perf /= n_trail
        print('Eval Loss {:0.4f}, performance {:0.4f}'.format(eval_loss, perf))
        return perf

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
    
    def eval_before(self, n_trial, task_end):
        for i in range(task_end+1):
            print(f'Eval task {i} after training task {task_end}')
            self.eval(self.datasets.evalsets[i], n_trial)

    def train_all(self, iter, lr, record_freq, ckp_dir, omega_dir, eval_freq=1000):
        for i in range(len(self.datasets.trainsets)):
            print('Training task', i)
            if i == 0:
                self.train(self.datasets.trainsets[i], self.datasets.evalsets[i], iter, lr, record_freq, ckp_dir, omega_dir, eval_freq=eval_freq, c=None)
                self.eval_before(100, i)
                continue
            self.train(self.datasets.trainsets[i], self.datasets.evalsets[i], iter, lr, record_freq, ckp_dir, omega_dir, eval_freq=eval_freq)
            self.eval_before(100, i)

        return self.model, self.Omega