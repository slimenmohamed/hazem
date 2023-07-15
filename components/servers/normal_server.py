import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
import pandas as pd
import numpy as np
from typing import OrderedDict
import torch
import torch.nn as nn
from utilities.evalutation_utilities import flatten, compute_accuracy, get_mean_accuracy_for_group
import gc
import copy
from tqdm import tqdm
import random
from utilities.logger_setup import create_logger
from utilities.data_utlities import number_of_classes_from_datasets
# TODO: fix the plotting
import matplotlib.pyplot as plt

class NormalServer:
    def __init__(self, config, data_dicts, test_data_dicts, hospitals_datasets, global_datasets, names, server_train_idxs, server_test_idxs) -> None:
        self.logger = create_logger(config["log_path"])
        self.config = config
        self.data_dicts = data_dicts
        self.test_dicts = test_data_dicts
        self.hospitals_datasets = hospitals_datasets
        self.global_datasets = global_datasets
        self.model = config["model"](config["num_classes"])
        self.names = names
        self.hospital_ids = [*data_dicts.keys()]
        self.participation_percent = config["participation_percent"]
        self.cluste2Frs = []
        self.sim = None
        self.num_classes = number_of_classes_from_datasets(names)
        ###### CHOOSING THE hospital ######
        self.hospital = config["node_type"]
        self.make_hospitals()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.similarity_matrices = []
        self.trained = []

          ##### Server train and test indices ######
        self.train_idxs = server_train_idxs
        self.test_idxs = server_test_idxs
        self.num_train_samples = sum([len(i) for i in self.train_idxs])
        self.global_results_dict = {}
        print("finished server intitialization")
        self.data_frame = pd.DataFrame(columns=["Accuracy", "Loss", "Sensitivity", "Specificity", "F1Score", "ValLoss"], index=list(range(config["global_epochs"]+1)))
        self.logits = []
        self.loss_fn = nn.CrossEntropyLoss()
        self.total_data_points_trained_on = 0
        self.total_dataset = []
        self.logger.info("Finished initializing server attributes")
        for i, (_, dataset) in enumerate(zip(self.names, self.global_datasets)):
          dataloader = DataLoader(dataset, batch_size=1, sampler=self.train_idxs[i])
          for x, y in dataloader:
             self.total_dataset.append((x.squeeze(0),y.squeeze(0)))
        self.logger.info("Finished creating the mixed set for the server")

    def make_hospitals(self):
        self.hospitals = []
        index = 0
        self.logger.info(f"Instanaciating hospitals")
        for i, hospital in enumerate(self.hospital_ids):
          self.hospitals.append(self.hospital(
              id=hospital,
              config=self.config,
              dataset=self.hospitals_datasets[index],
              data_idxs=self.data_dicts[hospital],
              test_idxs=self.test_dicts[hospital],
              ds_name=self.names[index]
              ))
          if i%10==9:
            index += 1
        self.logger.info(f"Created {len(self.hospitals)} hospital classes, IDS from {min(self.hospital_ids)} to {max(self.hospital_ids)}")


    def test_locals(self):
        results = {id: [] for id in self.hospital_ids}
        for hospital in self.hospitals:
            results[hospital.id] = hospital.test()
        return results

    def aggregate_(self, wstate_dicts):
        total_weight = sum(weight for weight, _ in wstate_dicts.values())
        aggregated_state_dict = {}

        for _, (weight, state_dict) in wstate_dicts.items():
            for key, value in state_dict.items():
                if key not in aggregated_state_dict:
                    aggregated_state_dict[key] = torch.zeros_like(value)

                aggregated_state_dict[key].data.add_(weight * value)

        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] = aggregated_state_dict[key].to(torch.float32) / total_weight

        return aggregated_state_dict

    def flatten_model(self, state_dict):
      flat = []
      for key in state_dict:
        flat.append(state_dict[key].flatten())
      return torch.cat(flat, dim=0)

    def test_hospitals_on_global(self, round_num, hospitals):
        results = {}
        for hospital_id in hospitals:
          true, pred = self.test_model_on_global_test_data(self.hospitals[hospital_id].model)
          accuracy =  compute_accuracy(true, pred)
          results[hospital_id] = {
            'accuracy': accuracy
          }

        self.global_results_dict[round_num] = results

    def aggregate_two_sd(self, sd1, sd2):
      result = {}
      for key, value in sd1.items():
        if key not in result:
          result[key] = torch.zeros_like(value)
        result[key].data.add_(value)
      for key, value in sd2.items():
        if key not in result:
          result[key] = torch.zeros_like(value)
        result[key].data.add_(value)
        result[key].data = result[key].data / 2
      return result

    def test_model_on_global_test_data(self, m=None):
        true = []
        preds = []
        accuracy = {}
        model = self.model if not m else m
        model.to(self.device)
        model.eval()
        with torch.no_grad():
          for i, (ds_name, dataset) in enumerate(zip(self.names, self.global_datasets)):
            dataloader = DataLoader(dataset, batch_size=32, sampler=self.test_idxs[i])
            correct = 0
            for x, y in dataloader:
              x = x.to('cuda')
              y = y.type(torch.LongTensor).to('cuda')
              logits = model(x)
              pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
              preds.append(pred.flatten())
              true.append(y.cpu().numpy().flatten())
              # print(preds.flatten(), y.flatten())
              correct += np.sum(np.array(flatten(preds)).flatten() == y.cpu().numpy().flatten())
            accuracy[ds_name]= correct/ len(dataset)
        if m == None:
          return accuracy
        return np.array(flatten(true)).flatten(), np.array(flatten(preds)).flatten()

    def train_prev_server_on_global(self, roundnum, epochs=4):
        previous_model = self.config["model"](self.num_classes)
        previous_model.to("cuda")
        previous_model.load_state_dict(self.previous_model_sd)
        self.previous_model = previous_model
        self.previous_model.train()
        self.previous_model.to(self.device)
        # old_parameters = self.model.parameters() / uncomment this for fedprox
        optimizer = SGD(self.previous_model.parameters(), self.config["learning_rate"])
        train_loss = []
        self.logits.append([])
        dataloader = DataLoader(self.total_dataset, batch_size=32, shuffle=True)
        for e in range(epochs):
          for x, y in dataloader:
            x = x.to(self.device)
            y = y.type(torch.LongTensor)
            y = y.to(self.device)
            optimizer.zero_grad()
            logits = self.previous_model(x)
            loss = self.loss_fn(logits, y.squeeze(-1)) # + (mu/2)*proximal_term / uncomment this for fedprox // , weight=self.class_weights
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().item())
            self.logits[-1].append(logits.detach().cpu())
            self.total_data_points_trained_on += len(y)
            # print(loss.item())
        self.last_train_loss = train_loss
        #self.avg_loss = np.mean(train_loss)
        self.data_frame.loc[roundnum, "Loss"] = np.mean(train_loss)
        return

    def train(self, k=None):
        epochs = self.config["global_epochs"]
        self.avg_loss_arr = [np.nan for _ in range(epochs)]
        self.avg_acc_arr = [np.nan for _ in range(epochs)]
        self.avg_loss_arr_std = [np.nan for _ in range(epochs)]
        self.upper = [np.nan for _ in range(epochs)]
        self.lower = [np.nan for _ in range(epochs)]
        ylim = 4
        self.sorted_hospitals = []
        # We start our normal training
        for i, epoch in tqdm(enumerate(range(1, epochs+1))):
            self.avg_loss = []
            # Optimization purposes
            self.previous_model_sd = copy.deepcopy(self.model.state_dict())
            torch.cuda.empty_cache()
            gc.collect()
            # Choosing a fraction of hospitals
            hospitals = random.sample([hospital.id for hospital in self.hospitals], int(len(self.hospitals)* self.participation_percent))
            self.states_dict = {}
            self.trained.append(hospitals)
            # We loop of the clusters, each cluster we train all the hospitals inside of it
            for hospital in hospitals:
                self.hospitals[hospital].model.load_state_dict(self.model.state_dict())
                w, update = self.hospitals[hospital].train(roundnum=epoch)
                self.states_dict[hospital] = [w, update]
                self.avg_loss.append(self.hospitals[hospital].avg_loss)

            self.test_hospitals_on_global(epoch, self.trained[-1])
            self.train_prev_server_on_global(epoch, epochs=5)
            self.model.to(self.device)
            self.previous_model.to(self.device)
            self.model.load_state_dict(
                  self.aggregate_two_sd(self.model.state_dict(), self.previous_model.state_dict())
                )

            acc, val_loss = self.test()
            self.data_frame.loc[epoch, "Accuracy"] = acc
            self.data_frame.loc[epoch, "ValLoss"] = val_loss
            ########### PLOTTING ###########

            # Plotting the loss
            self.avg_loss_arr[i] = np.mean(self.avg_loss)

            # Getting the upper and lower bounds of the std of the loss
            self.avg_loss_arr_std[i] = np.std(self.avg_loss)
            self.upper[i] = self.avg_loss_arr[i] + self.avg_loss_arr_std[i]
            self.lower[i] = self.avg_loss_arr[i] - self.avg_loss_arr_std[i]

            # updating the size of the loss plot
            ymax = max([*self.upper]) + 1
            ylim = max(ylim, ymax)
            # clear_output(wait=True)
            plt.figure(figsize=(9,4))
            plt.ylim(0, ylim)
            plt.xlim(0, epochs)

            # plotting loss and std
            plt.plot(list(range(epochs)), self.avg_loss_arr, color="C0", label="Loss")
            plt.fill_between(list(range(epochs)), self.lower, self.upper, alpha=0.5, color="C0", label="± std of loss")
            plt.grid(True, color='gray', linestyle='dashed', linewidth=0.5)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Average Loss for hospitals trained until round {epoch}")
            plt.legend(loc="upper right")
            plt.show()

            self.avg_acc_arr[i] = get_mean_accuracy_for_group(server=self, group=self.trained[-1], round=i)

            # plotting average test accuracy
            plt.plot(self.avg_acc_arr, label="Average test accuracy for trained hospitals at that round", color="orange")
            plt.grid(True, color='gray', linestyle='dashed', linewidth=0.5)
            plt.ylim(0,1)
            plt.xlim(0,epochs)
            plt.legend(loc="lower right")
            plt.show()
            self.global_accuracies = [np.mean([hospital['accuracy'] for hospital in group.values()]) for group in self.global_results_dict.values()]
            plt.plot(self.global_accuracies, label="Global accuracy")
            plt.grid(True, color='gray', linestyle='dashed', linewidth=0.5)
            plt.xlim(0,epochs)
            plt.ylim(0,1)
            plt.legend()
            plt.show()


    def test(self, train_before=False, raw=False):
        # one training pass before testing
        if train_before:
          optimizer = SGD(self.model.parameters(), self.config["learning_rate"])
          self.model.train()
          self.model.to(self.device)
          for x, y in self.train_loader:
              x = x.to(self.device)
              y = y.to(self.device)
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
          for i, (ds_name, dataset) in enumerate(zip(self.names, self.global_datasets)):
            dataloader = DataLoader(dataset, batch_size=32, sampler=self.test_idxs[i])
            correct = 0
            for x, y in dataloader:
              x = x.to('cuda')
              y = y.type(torch.LongTensor).to('cuda')
              logits = self.model(x)
              pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
              val_loss += self.loss_fn(logits ,y.flatten()).detach().cpu().item( )
              self.preds.append(pred.flatten())
              self.y.append(y.cpu().numpy().flatten())

        val_loss /= len(self.y)
        accuracy = compute_accuracy(np.array(flatten(self.y)), np.array(flatten(self.preds)))
        return accuracy, val_loss