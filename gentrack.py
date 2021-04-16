import math

import torch
import matplotlib.pyplot as plt
from Net import Net
from converter import convert_table_to_track, get_column_from_table_dict
from utils import draw_track


def gen_track(model, age=11465471475, device=torch.device("cpu")):
    log_L = []
    log_Teff = []

    model.to(device)

    track = []

    for i in range(1, age, 100000000):
        data = torch.Tensor([1, math.log10(i) / 2]).to(device)
        output = model(data).tolist()

        L = output[1]
        T = output[2]
        # if (0 > L > 10) or (0 > T > 10):
        #     print("Skip")
        #     continue
        track.append(output)

        log_L.append(L)
        log_Teff.append(T)

        print(output)
        print(i / age * 100, '%')
    return log_L, log_Teff

    # print(log_L)

    # plt.ion()


def compare_tracks(model, path, age=11465471475, device=torch.device("cpu")):
    track = convert_table_to_track(path)['track']
    x_orig = get_column_from_table_dict(track, 'log_Teff')
    y_orig = get_column_from_table_dict(track, 'log_L')

    x, y = gen_track(model, age, device)

    plt.plot(x_orig, y_orig, label='Original')
    plt.xlabel('log_Teff')
    plt.ylabel('log_L')

    plt.scatter(x, y, label='Generated')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()

# net = Net()
#
# net.load_state_dict(torch.load('model.pt'))
#
# net.eval()
# compare_tracks(model=net, path='datasets/tracks/0010000M.track.eep', device=torch.device("cuda"))
