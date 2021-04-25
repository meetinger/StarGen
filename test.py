import os

import torch
from torch import nn

from Net import Net
from converter import convert_table_to_track
from gentrack import compare_tracks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Net()

if torch.cuda.is_available():
    net = net.cuda(device)
    print("Cuda Available!")
else:
    print("Cuda Not Available!")

if os.path.isfile('model.pt'):
    net.load_state_dict(torch.load('model.pt', map_location=device))

net.eval()


path = 'datasets/tracks/0001000M.track.eep'


track = convert_table_to_track(path)

compare_tracks(model=net, path=path, device=device)