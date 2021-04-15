import math

import torch
import matplotlib.pyplot as plt
from Net import Net
from converter import convert_table_to_track
from utils import draw_track

model = Net()

model.load_state_dict(torch.load('model.pt'))

model.eval()

log_L=[]
log_Teff=[]

track = []
age = 11465471475
for i in range(1, age, 100000000):
    data = torch.Tensor([1, math.log10(i)])
    output = model(data).tolist()
    print(output)
    L = output[1]
    T = output[2]
    # if (0 > L > 10) or (0 > T > 10):
    #     print("Skip")
    #     continue
    track.append(output)

    log_L.append(L)
    log_Teff.append(T)
    print(i/age*100, '%')

# print(log_L)
plt.plot(log_L, log_Teff)
plt.xlabel('log_Teff')
plt.ylabel('log_L')
# plt.gca().invert_xaxis()

path = 'datasets/tracks/0010000M.track.eep'
track = convert_table_to_track(path)
draw_track(track)