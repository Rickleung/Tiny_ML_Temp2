import pickle
import torch
from torchsummary import summary


# Load the model
model = torch.load('/home/rick/Downloads/luis.pt', map_location=torch.device('cpu'))

# Print the model structure
print(model)

# # Load the .pkl file
# with open('/home/rick/Downloads/luis.pt', 'rb') as f:
#     data = torch.load(f, map_location=torch.device('cpu'))
#     summary(data, input_size=(channels, height, width))
# # Check the structure of the loaded data
# #print(data)  # Inspect the contents to locate the relevant information
#
# # Access and print the values if available
# if 'in_channels' in data:
#     in_channels = data['in_channels']
#     print(f'in_channels: {in_channels}')
#
# if 'out_channels' in data:
#     out_channels = data['out_channels']
#     print(f'out_channels: {out_channels}')
#
# if 'kernel_size' in data:
#     kernel_size = data['kernel_size']
#     print(f'kernel_size: {kernel_size}')
#
# if 'stride' in data:
#     stride = data['stride']
#     print(f'stride: {stride}')
#
