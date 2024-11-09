import sys
from ftplib import all_errors

import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader, random_split
from utilities.priority_queue import *


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

class TrajectoryDataset(Dataset):
    def __init__(self, data, buffer_indices):
        self.data = data
        # saves buffer indices also for updating relevant indices in the tree after sampling from buffer
        self.buffer_indices = buffer_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx][0]  # State vector (tensor)
        action = self.data[idx][1]  # Action (tensor)
        cost = self.data[idx][2]  # Cost (tensor)
        buffer_idx = self.buffer_indices[idx]
        input_features = torch.cat((state, action.float()))  # Concat state and action
        return input_features, cost.float(), buffer_idx



class Shield(nn.Module):
    def __init__(self, input_size, loss_fn, lr, k_epochs, batch_size, buffer_size, sub_batch_size):
        super(Shield, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)  # Output: single value (cost)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.parameters(),  lr)
        self.buffer_size = buffer_size
        self.buffer = PER_Buffer(self.buffer_size)
        self.batch_size = batch_size
        self.k_epochs = k_epochs
        self.sub_batch_size = sub_batch_size

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # No activation, since it's regression
        return x

    def loss(self, predictions, costs):
        loss = self.loss_fn(predictions, costs)
        return loss

    def update(self, num_update):
        buffer_len = self.buffer.get_buffer_len()
        if buffer_len == 0:
            return 0
        elif buffer_len <= self.batch_size:
            batch_size = buffer_len
        else:
            batch_size = self.batch_size
        batch_samples, idxs, is_weight = self.buffer.sample(batch_size)
        batch_samples_states = torch.stack([sample[0] for sample in batch_samples])
        batch_samples_actions = torch.stack([sample[1] for sample in batch_samples])
        batch_samples_costs = torch.stack([sample[2] for sample in batch_samples]).to(device)
        # prepare input_features and label (cost) for supervised learning
        input_features_and_label = TrajectoryDataset(batch_samples, idxs)
        data_loader = DataLoader(input_features_and_label, batch_size=self.sub_batch_size, shuffle=True)
        total_loss = 0
        all_buffer_indices = []
        all_errors = []
        for ind, (inputs, targets, buffer_idxs) in enumerate(data_loader):
            inputs, targets = inputs.float().to(device), targets.float().view(-1,1).to(device)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.loss(outputs, targets.view(-1,1))
            loss.backward()
            buffer_errors = (outputs - targets).abs().cpu().detach().numpy()
            all_buffer_indices.extend(buffer_idxs.tolist())
            all_errors.extend(buffer_errors)
            self.optimizer.step()
            total_loss += loss.item()
        # udpate the buffer based on new errors for sampled data
        for buffer_idx, error in zip(all_buffer_indices, all_errors):
            self.buffer.update(buffer_idx, torch.tensor(error))
        avg_loss = total_loss / len(data_loader)
        return avg_loss

# calculation of error for buffer: episode_errors = torch.abs(predictions - costs_ten).cpu().data.numpy()
        """
        buffer_pkl_filename = "models/03.11/action_selection_method/random/update_samples_pickles" + f"/{num_update}_agent_buffer_samples.pkl"

        with open(buffer_pkl_filename, "wb") as f:
            pickle.dump(batch_samples, f)
            print(f"Agent buffer saved to {buffer_pkl_filename} successfully with {len(batch_samples)} samples")
        """





    """
    4.11 
    
    def update(self, num_update):
        buffer_len = self.buffer.get_buffer_len()
        if buffer_len == 0:
            return 0
        elif buffer_len <= self.batch_size:
            batch_size = buffer_len
        else:
            batch_size = self.batch_size

        # Sample buffer
        batch_samples, idxs, is_weight = self.buffer.sample(batch_size)
        batch_samples_states = torch.stack([sample[0] for sample in batch_samples])
        batch_samples_actions = torch.stack([sample[1] for sample in batch_samples])
        batch_samples_costs = torch.stack([sample[2] for sample in batch_samples]).to(device)

        buffer_pkl_filename = "models/03.11/action_selection_method/random/update_samples_pickles" + f"/{num_update}_agent_buffer_samples.pkl"

        with open(buffer_pkl_filename, "wb") as f:
            pickle.dump(batch_samples, f)
            print(f"Agent buffer saved to {buffer_pkl_filename} successfully with {len(batch_samples)} samples")


        # Prepare model input (Combine state and action into one tensor)
        x = torch.cat([batch_samples_states, batch_samples_actions], -1).to(device)
        x = x.float()

        # Sub-batch parameters
        sub_batch_size = min(self.sub_batch_size, batch_size)  # Define a sub_batch_size in your class
        num_sub_batches = int(batch_size / sub_batch_size)

        total_epochs_loss = 0

        # Iterate over sub-batches and update the model
        for sub_batch_idx in range(num_sub_batches):
            # Randomly sample a sub-batch from the full batch
            sub_idx = torch.randint(0, batch_size, (sub_batch_size,))
            sub_batch_states = x[sub_idx]
            sub_batch_costs = batch_samples_costs[sub_idx]

            self.optimizer.zero_grad()
            predictions = self.forward(sub_batch_states).squeeze(1)
            # Compute the loss for the current sub-batch
            loss = self.loss(predictions, sub_batch_costs)
            loss.backward()
            self.optimizer.step()

            total_epochs_loss += loss.item()

        # TODO - UPDATE THE SAMPLES IN THE BUFFER

        return total_epochs_loss / num_sub_batches
    """
    def add_to_buffer(self, error, state_action):
        self.buffer.add(error, state_action)

