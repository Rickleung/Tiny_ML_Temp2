import torch
import pickle

model = torch.load('/home/rickleung/Downloads/luis.pt', map_location=torch.device('cpu'))
with open('/home/rickleung/Downloads/luis.pkl', 'wb') as f:
    pickle.dump(model, f)
