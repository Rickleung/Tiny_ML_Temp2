# import torch
#
# # Open the file in read-binary mode
# with open('/home/rickleung/Downloads/luis.pt', 'rb') as f:
#     # Load the model
#     model = torch.load(f, map_location=torch.device('cpu'))
#
# # Print the model structure
# print(model)

import pandas as pd

data = pd.read_pickle(r'/home/rickleung/Downloads/luis.pkl')
print(data)
