import torch
import pickle

model = torch.load('/home/rick/Downloads/luis.pt', map_location=torch.device('cpu'))
with open('/home/rick/Downloads/luis.pkl', 'wb') as f:
    pickle.dump(model, f)
