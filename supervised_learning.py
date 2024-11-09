#to create dirs
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import random
import torch.nn.functional as F  # Import functional for MSE calculation
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna

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



# Model Training

# Manually define arguments (not from commandArgs)
# 36
update_samples_paths =  ["models/03.11/action_selection_method/random/update_samples_pickles/" + f"{i}_agent_buffer_samples.pkl" for i in range(52)]
lr_shield = 0.001
batch_size = 500
num_epochs = 50
random_seed = 0

#print("random seed is set to ", random_seed)
torch.manual_seed(random_seed)
# CUDA seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# need to manually change it
architecture = "new"

base_path = f"models/03.11/action_selection_method/random/supervised_learning/5-layers/lr_shield={lr_shield}/"
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




# Define the dataset class
class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx][0]  # State vector (tensor)
        action = self.data[idx][1]  # Action (tensor)
        cost = self.data[idx][2]  # Cost (tensor)
        input_features = torch.cat((state, action.float()))  # Concat state and action
        return input_features, cost.float()


# prepare the data - train-test split

train_trajectories = all_trajectories[:-1]
test_trajectories = all_trajectories[-1]

train_datasets = [TrajectoryDataset(trajectories) for trajectories in train_trajectories]
test_dataset = TrajectoryDataset(test_trajectories)

train_loaders = [DataLoader(traj, batch_size=batch_size, shuffle=True) for traj in train_datasets]
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Old Architecture
"""
class Shield(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space):
        super().__init__()
        hidden_dim = 64
        self.has_continuous_action_space = has_continuous_action_space
        if not self.has_continuous_action_space:
            self.action_embedding = nn.Embedding(action_dim, action_dim)
            self.action_embedding.weight.data = torch.eye(action_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        #self.loss_fn = torch.nn.SmoothL1Loss()
        self.loss_fn = torch.nn.MSELoss()

    def encode_action(self, a):
        # returns one hot vector with the given action - for example (0,1) for action 2
        if self.has_continuous_action_space:
            return a
        else:
            one_hot = torch.zeros(a.size(0), 2).to(device)
            one_hot.scatter_(1, a.long(), 1)
            return one_hot.to(device)

    def forward(self, s, a):
        # pass (state,action)) in the network. returns the unsafe score.
        a = self.encode_action(a)
        x = torch.cat([s, a], -1).to(device)
        return self.net(x)

    def loss(self, predictions, costs):
        loss = self.loss_fn(predictions.to(device), costs.to(device))
        return loss
"""

# New Architecture
class Shield(nn.Module):
    def __init__(self, input_size):
        super(Shield, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)  # Output: single value (cost)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # No activation, since it's regression
        return x


#input_size = 4 + 1  # CartPole state is 4-dimensional, plus 1 action (not using one-hot encoding)

# OLD Architecture
#shield_model = Shield(4, action_dim = 2, has_continuous_action_space = False).to(device)

# New Architecture
shield_model = Shield(input_size=5).to(device)


# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(shield_model.parameters(), lr = lr_shield)

# Training loop

loss_epochs_values = []

# Training loop
for train_loader in train_loaders: # // equivalent to batch sampled from shield buffer in shield_net.update()
    total_loss = 0
    for ind, (inputs, targets) in enumerate(train_loader): # // equivalent to sub-batches in shield_net.update()
        inputs = inputs.float()  # Ensure inputs are Float
        inputs, targets = inputs.to(device), targets.float().view(-1, 1).to(device)
        optimizer.zero_grad()  # Clear gradients

        # Old Architecture
        """
        state_ = inputs[:, :4]  # First 4 columns
        action_ = inputs[:, 4:]  # Last column (which will have shape (1024, 1))
        outputs = shield_model(state_.to(device),action_.to(device))  # Forward pass
        """
        # New Architecture
        outputs = shield_model(inputs)  # Forward pass
        loss = criterion(outputs, targets.view(-1, 1))  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        total_loss += loss.item()  # Accumulate loss
    #    print(f'Batch Number [{ind + 1} for epoch {epoch+1}], Loss: {loss:.6f}')

    avg_epoch_loss = total_loss / len(train_loader)  # Average loss for the epoch
    loss_epochs_values.append(avg_epoch_loss)  # Store the average loss for this epoch

loss_df = pd.DataFrame(loss_epochs_values, columns=['Shield Loss'])
csv_path = base_path + f"loss_values_num_epochs={num_epochs}.csv"
model_path = base_path + "trained_shield.pth"
loss_df.to_csv(csv_path, index=False)
torch.save(shield_model.state_dict(), model_path)

# Excel that shows the labels 'distribution'. (as floats)

"""
costs = [dat[2] for dat in trajectories]

# Step 1: Get unique values and their counts
costs_series = pd.Series(costs)
unique_costs_counts = costs_series.value_counts().reset_index()
unique_costs_counts.columns = ['Cost', 'Count']  # Rename columns

# Step 2: Save to an Excel file
unique_costs_counts.to_csv("costs_unique_values.csv", index=False)
shield_trained_model = "models/save_buffer/250k/trained_shield.pth"
"""

# Evaluate the model using PyTorch's MSE
total_mse = 0
total_samples = 0

predictions_list = []
targets_list = []

with torch.no_grad():  # Disable gradient computation for testing
    for inputs, targets in test_loader:
        inputs, targets = inputs.float().to(device), targets.float().view(-1, 1).to(device)

        # Old Architecture
        """xcopy /E /I shira_env xcopy /E /I shira_env 
        state_ = inputs[:, :4]  # First 4 columns
        action_ = inputs[:, 4:]  # Last column (which will have shape (1024, 1))
        outputs = shield_model(state_.to(device),action_.to(device))  # Forward pass
        """
        # New Architecture
        outputs = shield_model(inputs)

        # Collect predictions and targets for saving
        predictions_list.extend(outputs.cpu().numpy())
        targets_list.extend(targets.cpu().numpy())

        # Calculate MSE using PyTorch
        mse = F.mse_loss(outputs, targets)  # Mean Squared Error
        total_mse += mse.item() * inputs.size(0)  # Accumulate MSE weighted by batch size
        total_samples += inputs.size(0)  # Count total samples


# Compute average MSE
average_mse = total_mse / total_samples
print(f'Average Mean Squared Error (MSE): {average_mse:.6f}')

results_df = pd.DataFrame({
    'Prediction': np.array(predictions_list).flatten(),
    'Cost': np.array(targets_list).flatten()
})

# Step 1: Read the CSV file
loss_df = pd.read_csv(base_path + f"loss_values_num_epochs={num_epochs}.csv")

epochs = loss_df.index  # This will give you an implicit epoch number (0, 1, 2, ..., 49)
loss_values = loss_df['Shield Loss']  # Get the loss values

# Step 3: Plot the loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', label='Shield Loss')
plt.title('Shield Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Shield Loss')
plt.xticks(epochs)  # Set x-ticks to be the epochs for clarity
plt.grid(True)
plt.legend()


results_csv_path = os.path.join(base_path, 'test_predictions_vs_costs.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"Predictions and costs saved to {results_csv_path}")

# Step 4: Save the plot (optional)
plt.savefig(base_path + f'shield loss over time.png')
