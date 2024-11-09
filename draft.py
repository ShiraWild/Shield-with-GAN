

import pickle

# Replace 'your_file.pkl' with the path to your pickle file
with open('models/250k/lr_shield=5e-05/shield_buffer_samples.pkl', 'rb') as file:
    data = pickle.load(file)

# Check the length
try:
    print("Length of the data:", len(data))
except TypeError:
    print("Data does not support length checking with len().")
