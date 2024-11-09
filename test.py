import torch
print(torch.__version__)  # Check PyTorch version
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.cuda.current_device())  # Check the current device
print(torch.cuda.get_device_name(0))  # Get the name of the GPU