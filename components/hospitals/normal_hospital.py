from medmnist import INFO
from torch.utils.data import DataLoader
from torch.optim import SGD
import pandas as pd
import numpy as np
from typing import OrderedDict
import torch
import torch.nn as nn

####custom imports#####
from utilities.evalutation_utilities import calculate_hospital_score, flatten, compute_accuracy
from utilities.logger_setup import create_logger
class NormalHospital:
    def __init__(self, id, config, dataset, data_idxs, test_idxs, ds_name) -> None:
        self.id = id
        self.config = config
        self.train_loader = DataLoader(dataset, sampler=data_idxs, batch_size=config["batch_size"])
        self.test_loader = DataLoader(dataset, sampler=test_idxs, batch_size=32)
        self.device = config["device"]
        self.model = config["model"](config["num_classes"])
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_train_samples = len(data_idxs)
        self.ds_name = ds_name
        self.total_data_points_trained_on = 0
        self.data_frame = pd.DataFrame(columns=["Accuracy", "Loss", "ValLoss"], index=list(range(config["global_epochs"]+1)))
        self.num_times_chosen = 0
        self.score = np.random.rand()
        self.logger = create_logger(self.config["log_path"])

    def train(self, roundnum):
        self.num_times_chosen += 1
        optimizer = SGD(self.model.parameters(), self.config["learning_rate"])
        self.model.train()
        self.model.to(self.device)
        # old_parameters = self.model.parameters() / uncomment this for fedprox
        train_loss = []
        for local_epoch in range(self.config["local_epochs"]):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.type(torch.LongTensor).to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                # / uncomment this for fedprox
                # proximal_term = 0.0
                # iterate through the current and global model parameters
                # for w, w_t in zip(self.model.parameters(), old_parameters):
                #     proximal_term += (w-w_t).norm(2)

                loss = self.loss_fn(logits, y.squeeze(-1)) # + (mu/2)*proximal_term / uncomment this for fedprox
                loss.backward()
                optimizer.step()
                train_loss.append(loss.detach().item())
                self.total_data_points_trained_on += len(y)
        self.last_train_loss = train_loss
        self.score = self.compute_score()
        self.avg_loss = np.mean(train_loss)
        acc, val_loss = self.test()
        self.data_frame.loc[roundnum, "Loss"] = np.mean(train_loss)
        self.data_frame.loc[roundnum, "Accuracy"] = acc
        self.data_frame.loc[roundnum, "ValLoss"] = val_loss

        return self.num_train_samples, self.model.state_dict()

    def compute_score(self):
        # self.logger.debug(self.last_train_loss)
        return calculate_hospital_score(self.last_train_loss)

    def get_param(self) -> OrderedDict:
        return self.model.state_dict()

    def set_param(self) -> bool:
        self.model.load_state_dict()
        return True

    def test(self, train_before=False, raw=False):
        # one training pass before testing
        if train_before:
          optimizer = SGD(self.model.parameters(), self.config["learning_rate"])
          self.model.train()
          self.model.to(self.device)
          for x, y in self.train_loader:
              x = x.to(self.device)
              y = y.type(torch.LongTensor).to(self.device)
              optimizer.zero_grad()
              logits = self.model(x)
              loss = self.loss_fn(logits, y.squeeze(-1))
              loss.backward()
              optimizer.step()
        self.model.to(self.device)
        total_correct = 0
        self.model.eval()
        self.preds = []
        self.y = []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x=x.to(self.device)
                y=y.type(torch.LongTensor).to(self.device)
                logits = self.model(x)
                val_loss += self.loss_fn(logits ,y.flatten()).detach().cpu().item( )
                preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
                correct = np.equal(preds.flatten(), y.detach().cpu().flatten().numpy())
                correct = np.sum(correct)
                total_correct += correct
                self.preds.append(preds)
                self.y.append(y.detach().cpu().flatten().numpy())
        val_loss /= len(self.y)
        accuracy = compute_accuracy(np.array(flatten(self.y)), np.array(flatten(self.preds)))
        return accuracy, val_loss