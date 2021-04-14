import random

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def clean_split(str, delimiter, ex_chars=None):
    if ex_chars is None:
        ex_chars = ['']
    splitted = str.split(delimiter)
    cleaned = [s.strip() for s in splitted if not (s in ex_chars)]
    return cleaned


def read_table(path, header_line, delimiter=" ", header_params=None, max_line=None):
    # with open(path, 'r') as table:
    #     for _ in range(header):
    #         table.next()

    table = open(path, 'r').readlines()
    # print(table)
    # header = table[header_line - 1].split(delimiter)
    # header = [s.strip() for s in header if (s != '') and (s != '#')]

    header_orig = clean_split(table[header_line], delimiter=' ', ex_chars=['', '#'])

    header = header_orig

    header_indexes = list(range(0, len(header)))
    # print(header)
    # print(header_indexes)
    if not (header_params is None):
        header = header_params
        header_indexes = [i for i in range(len(header_orig)) if header_orig[i] in header]

    # header_dict = {i: for i in header}

    header_dict = dict(zip(header, header_indexes))

    # print(header_dict)

    # print(header)
    # print(header_indexes)

    # values = { i :[] for i in header}
    # print(values)

    res = []

    if max_line is None:
        max_line = len(table)

    for i in range(header_line + 1, max_line):
        line = clean_split(table[i], delimiter=' ', ex_chars=['', '#'])
        tmp = dict.fromkeys(header)
        for key in header:
            tmp[key] = line[header_dict[key]]
        res.append(tmp)
    return res


def convert_table_to_track(path, params=None, initial_params=None):
    if initial_params is None:
        initial_params = ['initial_mass']
    if params is None:
        params = ['star_age', 'star_mass', 'log_L', 'log_Teff', 'phase']

    raw_data = read_table(path, header_line=11, delimiter=" ", header_params=params)

    tmp_initial_data = []

    if initial_params:
        tmp_initial_data = read_table(path, header_line=6, delimiter=" ", header_params=initial_params, max_line=8)[0]

    initial_data = {i: float(tmp_initial_data[i]) for i in tmp_initial_data}

    tmp_data = []

    for line in raw_data:
        tmp = {i: float(line[i]) for i in line}
        tmp_data.append(tmp)

    data = {'initial_params': initial_data, 'track': tmp_data}

    return data
    # return data


def get_column_from_table_dict(data, key):
    tmp = [line[key] for line in data]
    return tmp


def create_dataset(data, dataset_obj=True):
    initial_params = data['initial_params']
    track = data['track']

    x = [(initial_params['initial_mass'], i['star_age']) for i in track]

    y = [tuple(i.values())[1::] for i in track]

    # if(dataset_obj):
    #     tensor_x = torch.Tensor(x)
    #     tensor_y = torch.Tensor(y)
    #
    #     dataset = TensorDataset(tensor_x, tensor_y)
    #
    #     # dataloader = DataLoader(dataset)
    #     return dataset
    # else:
    #     return x, y
    return x, y

def split_dataset(full, amount=0.8):
    full_size = len(full)

    train_size = int(amount * full_size)
    test_size = full_size - train_size

    indexes = list(range(len(full)))
    random.shuffle(indexes)

    train_data = []
    test_data = []

    for i in indexes:
        if i < train_size:
            train_data.append(full[i])
        else:
            test_data.append(full[i])

    return (train_data, test_data)