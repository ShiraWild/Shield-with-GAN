import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import random
import torch.nn.functional as F
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import time

################################## set device ##################################
print("============================================================================================")
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

# Manually define arguments (not from commandArgs)
update_samples_paths = [
    "models/03.11/action_selection_method/random/update_samples_pickles/" + f"{i}_agent_buffer_samples.pkl" for i in
    range(6)]
random_seed = 0

torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

architecture = "new"
base_path = f"models/03.11/action_selection_method/random/supervised_learning/5-layers/"
os.makedirs(base_path, exist_ok=True)

# Load the samples
all_trajectories = []
for samples_path in update_samples_paths:
    try:
        with open(samples_path, 'rb') as file:
            trajectories = pickle.load(file)
            print(f"Successfully loaded  'samples_path' with {len(trajectories)} samples.")
            all_trajectories.append(trajectories)
    except Exception as e:
        print(f"An error occurred: {e}")


class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx][0]
        action = self.data[idx][1]
        cost = self.data[idx][2]
        input_features = torch.cat((state, action.float()))
        return input_features, cost.float()


train_trajectories = all_trajectories[:-1]
test_trajectories = all_trajectories[-1]

train_datasets = [TrajectoryDataset(trajectories) for trajectories in train_trajectories]
test_dataset = TrajectoryDataset(test_trajectories)


class Shield(nn.Module):
    def __init__(self, input_size):
        super(Shield, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def objective(trial):
    start_time = time.time()

    lr_shield = trial.suggest_loguniform('lr_shield', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 500])
    num_epochs = trial.suggest_int('num_epochs', 10, 100)

    shield_model = Shield(input_size=5).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(shield_model.parameters(), lr=lr_shield)

    train_loaders = [DataLoader(traj, batch_size=batch_size, shuffle=True) for traj in train_datasets]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        shield_model.train()
        total_loss = 0
        for train_loader in train_loaders:
            for inputs, targets in train_loader:
                inputs = inputs.float().to(device)
                targets = targets.float().view(-1, 1).to(device)
                optimizer.zero_grad()
                outputs = shield_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        avg_epoch_loss = total_loss / len(train_loader)

    shield_model.eval()
    total_mse = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().view(-1, 1).to(device)
            outputs = shield_model(inputs)
            mse = F.mse_loss(outputs, targets)
            total_mse += mse.item() * inputs.size(0)
            total_samples += inputs.size(0)
    average_mse = total_mse / total_samples

    end_time = time.time()
    trial_duration = end_time - start_time
    print(f"Trial {trial.number} took {trial_duration:.2f} seconds.")

    return average_mse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best hyperparameters: ', study.best_params)
print('Best score for relevant hyperpamaters: ', study.best_value)