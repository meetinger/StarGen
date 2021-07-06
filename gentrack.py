import math
import time

import torch
import matplotlib.pyplot as plt
from Net import Net
from converter import convert_table_to_track, get_column_from_table_dict, scale_age, unscale_output, scale_input, \
    phase_to_str, load_files
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



def interpolate(path, mass):
    paths = load_files(path)
    masses = list(paths.keys())
    print(masses)

    mass_a, mass_b = masses[0], masses[1]
    for i in range(1, len(masses)):
        if masses[i - 1] <= mass <= masses[i]:
            mass_a, mass_b = masses[i - 1], masses[i]

    if mass_a == mass:
        mass_a = masses[masses.index(mass_a) - 1]

    print(mass_a, mass_b)

    track_a = convert_table_to_track(paths[mass_a])['track']

    track_b = convert_table_to_track(paths[mass_b])['track']

    abs_a = (mass - mass_a)
    abs_a_b = (mass_b - mass_a)

    k_mass = abs_a/abs_a_b

    def linear_scale(xmin, xmax, k):
        return abs(xmax - xmin) * k + xmin

    max_len = min(len(track_a), len(track_b))

    print(max_len)

    interpolated = []

    for i in range(0, max_len):
        cur_a = track_a[i]
        cur_b = track_b[i]

        star_age = linear_scale(cur_a['star_age'], cur_b['star_age'], k_mass)
        star_mass = linear_scale(cur_a['star_mass'], cur_b['star_mass'], k_mass)
        log_L = linear_scale(cur_a['log_L'], cur_b['log_L'], k_mass)
        log_Teff = linear_scale(cur_a['log_Teff'], cur_b['log_Teff'], k_mass)
        phase = int(linear_scale(cur_a['phase'], cur_b['phase'], k_mass))

        tmp = {
            'star_age': star_age,
            'star_mass': star_mass,
            'log_L': log_L,
            'log_Teff': log_Teff,
            'phase': phase
        }

        interpolated.append(tmp)

    star_age = get_column_from_table_dict(interpolated, 'star_age')
    star_mass = get_column_from_table_dict(interpolated, 'star_mass')
    log_L = get_column_from_table_dict(interpolated, 'log_L')
    log_Teff = get_column_from_table_dict(interpolated, 'log_Teff')
    phase = get_column_from_table_dict(interpolated, 'phase')

    smoothed = []

    smooth_period = 50

    #moving average
    for i in range(0, max_len+smooth_period):
        slice_bound_left = max(1, i-smooth_period)
        slice_bound_right = min(max(i, 2), max_len)
        # exp = (slice_bound_right - slice_bound_left) <= smooth_period < i
        # exp = slice_bound_right-slice_bound_left <= smooth_period
        # if exp:
        #     slice_bound_left = i
        #     slice_bound_right = max_len
        slice_size = slice_bound_right-slice_bound_left
        print(slice_bound_left, slice_bound_right, slice_size)
        star_age_tmp = star_age[slice_bound_left:slice_bound_right]
        star_mass_tmp = star_mass[slice_bound_left:slice_bound_right]
        log_L_tmp = log_L[slice_bound_left:slice_bound_right]
        log_Teff_tmp = log_Teff[slice_bound_left:slice_bound_right]
        tmp = {
            'star_age': sum(star_age_tmp)/slice_size,
            'star_mass': sum(star_mass_tmp)/slice_size,
            'log_L': sum(log_L_tmp)/slice_size,
            'log_Teff': sum(log_Teff_tmp)/slice_size,
            'phase': phase[min(i, max_len-1)]
        }
        smoothed.append(tmp)

    # arithmetical mean
    # for i in range(smooth_period, max_len, smooth_period):
    #     slice_bound_left = i-smooth_period
    #     slice_bound_right = i
    #     if i > (max_len - smooth_period):
    #         slice_bound_right = max_len
    #         slice_bound_left = i
    #     slice_size = slice_bound_right-slice_bound_left
    #     print(slice_bound_left, slice_bound_right, slice_size)
    #     star_age_tmp = star_age[slice_bound_left:slice_bound_right]
    #     star_mass_tmp = star_mass[slice_bound_left:slice_bound_right]
    #     log_L_tmp = log_L[slice_bound_left:slice_bound_right]
    #     log_Teff_tmp = log_Teff[slice_bound_left:slice_bound_right]
    #     tmp = {
    #         'star_age': sum(star_age_tmp)/slice_size,
    #         'star_mass': sum(star_mass_tmp)/slice_size,
    #         'log_L': sum(log_L_tmp)/slice_size,
    #         'log_Teff': sum(log_Teff_tmp)/slice_size,
    #         'phase': phase[i]
    #     }
    #     smoothed.append(tmp)

    x_a = get_column_from_table_dict(track_a, 'log_Teff')
    y_a = get_column_from_table_dict(track_a, 'log_L')

    x_b = get_column_from_table_dict(track_b, 'log_Teff')
    y_b = get_column_from_table_dict(track_b, 'log_L')

    x_interpolated = get_column_from_table_dict(interpolated, 'log_Teff')
    y_interpolated = get_column_from_table_dict(interpolated, 'log_L')

    x_smoothed = get_column_from_table_dict(smoothed, 'log_Teff')
    y_smoothed = get_column_from_table_dict(smoothed, 'log_L')

    track_orig = convert_table_to_track('datasets/tracks/0010000M.track.eep')
    track = track_orig['track']
    x_orig = get_column_from_table_dict(track, 'log_Teff')
    y_orig = get_column_from_table_dict(track, 'log_L')
    # plt.scatter(x_orig, y_orig, label='Orig', color='orange')

    # plt.plot(x_a, y_a, label='A', color='blue')
    # plt.plot(x_b, y_b, label='B', color='red')
    plt.plot(x_interpolated, y_interpolated, label='Interpolated', color='green')
    plt.plot(x_smoothed, y_smoothed, label='Smoothed', color='pink')



    plt.xlabel('log_Teff')
    plt.ylabel('log_L')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()

    # print(a, b)
    return interpolated


def compare_tracks(model, path, age=11465471475, device=torch.device("cpu"), draw_phases=True, datascaling=False):
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


interpolate('datasets/tracks', 1)


# net = Net()
#
# net.load_state_dict(torch.load('model.pt'))
#
# net.eval()
# compare_tracks(model=net, path='datasets/tracks/0010000M.track.eep', device=torch.device("cuda"))
