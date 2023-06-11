import pickle
import torch
from torchsummary import summary


# # Load the model
# model = torch.load('/home/rickleung/Downloads/luis.pt', map_location=torch.device('cpu'))
#
# # Print the model structure
# print(model)


try:
    # Load the pickle file
    with open('/home/rickleung/Downloads/luis.pkl', 'rb') as f:
        data = pickle.load(f)
except (EOFError, pickle.UnpicklingError):
    print('An error occurred while loading the pickle file')
    data = None

# Check if the loaded data is a dictionary
if isinstance(data, dict):
    # Print all keys in the dictionary
    print(data.keys())
else:
    print('The data is not a dictionary')

#
# try:
#     # Load the pickle file
#     with open('/home/rickleung/PycharmProjects/pythonProject/Tiny_ML_Temp/assets/mcu_models/mbv2-w0.35/archive/'
#               'data.pkl', 'rb') as f:
#         data = pickle.load(f)
# except (EOFError, pickle.UnpicklingError):
#     print('An error occurred while loading the pickle file')
#     data = None
#
# # Check if the loaded data is a dictionary
# if isinstance(data, dict):
#     # Print all keys in the dictionary
#     print(data.keys())
# else:
#     print('The data is not a dictionary')
#
# data2 = data['first_conv']
# data['blocks'] = data.blocks
# data['feature_mix'] = data.feature_mix
# data['classifier'] = data.classifier

# print('first_conv', data2)
# print(data['blocks'])
# print(data['feature_mix'])
# print(data['classifier'])


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
