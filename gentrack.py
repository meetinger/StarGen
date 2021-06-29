import math

import torch
import matplotlib.pyplot as plt
from Net import Net
from converter import convert_table_to_track, get_column_from_table_dict, scale_age, unscale_output, scale_input, \
    phase_to_str
from utils import draw_track


def gen_track(model, age=11465471475, mass=1, device=torch.device("cpu"), step=100000000, datascaling=True):
    ages = []
    last_age = 0
    if type(age) is int:
        ages = range(1, age, step)
        last_age=age
    else:
        ages = age
        last_age = age[-1]

    log_L = []
    log_Teff = []
    phase = []



    model.eval()
    model.to(device)

    track = []
    # counter = 0
    for i in range(0, len(ages)):
        # for i in ages:
        #     data = torch.Tensor([1, math.log10(ages[i]) / 2]).to(device)
        #     data = torch.Tensor([1, ages[i]]).to(device)

        if datascaling:
            data = torch.Tensor(scale_input([mass, ages[i]])).to(device)
            output = unscale_output(model(data).tolist())
        else:
            data = torch.Tensor([[mass, scale_age(ages[i], last_age)]]).to(device)
            output = model(data).tolist()



        L = output[1]
        T = output[2]
        phs = output[3]
        # if (0 > L > 10) or (0 > T > 10):
        #     print("Skip")
        #     continue
        track.append(output)

        log_L.append(L)
        log_Teff.append(T)
        phase.append(phs)
        # print(data)
        # print(output)
        if i % 10 == 0:
            print(i / len(ages) * 100, '%')
    return log_L, log_Teff, phase

    # print(log_L)

    # plt.ion()


def compare_tracks(model, path, age=11465471475, device=torch.device("cpu"), draw_phases=True, datascaling = False):
    data = convert_table_to_track(path)
    track = data['track']
    mass = data['initial_params']['initial_mass']
    print(mass)
    x_orig = get_column_from_table_dict(track, 'log_Teff')
    y_orig = get_column_from_table_dict(track, 'log_L')
    phase_orig = get_column_from_table_dict(track, 'phase')

    y, x, phase = gen_track(model=model, age=get_column_from_table_dict(track, 'star_age'), mass=mass, device=device, datascaling=datascaling)

    str_phase_orig = list(map(phase_to_str, phase_orig))
    str_phase = list(map(phase_to_str, phase_orig))

    # print(str_phase_orig)
    # print(str_phase)


    plt.scatter(x_orig, y_orig, label='Original', color='blue')
    plt.xlabel('log_Teff')
    plt.ylabel('log_L')


    plt.scatter(x, y, label='Generated', color='red')
    plt.legend()
    plt.gca().invert_xaxis()

    if draw_phases:
        last_phase = ""
        for i in range(0, len(x)):
            if last_phase != str_phase[i]:
                plt.text(x[i], y[i], str_phase[i], color='red', bbox=dict(facecolor='white', edgecolor='red'))
                last_phase = str_phase[i]
        last_phase = ""
        for i in range(0, len(x_orig)):
            if last_phase != str_phase_orig[i]:
                plt.text(x_orig[i], y_orig[i], str_phase_orig[i], color='blue',bbox=dict(facecolor='white', edgecolor='blue'))
                last_phase = str_phase_orig[i]

    plt.show()

# net = Net()
#
# net.load_state_dict(torch.load('model.pt'))
#
# net.eval()
# compare_tracks(model=net, path='datasets/tracks/0010000M.track.eep', device=torch.device("cuda"))
