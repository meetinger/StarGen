import pandas as pd

def clean_split(str, delimiter, ex_chars=None):
    if ex_chars is None:
        ex_chars = ['']
    splitted = str.split(delimiter)
    cleaned = [s.strip() for s in splitted if not(s in ex_chars)]
    return cleaned

def read_table(path, header_line, delimiter=" ", header_params=None):
    # with open(path, 'r') as table:
    #     for _ in range(header):
    #         table.next()
    table = open(path, 'r').readlines()
    # print(table)
    # header = table[header_line - 1].split(delimiter)
    # header = [s.strip() for s in header if (s != '') and (s != '#')]

    header_orig = clean_split(table[header_line - 1], delimiter=' ', ex_chars=['', '#'])

    header = header_orig

    header_indexes = list(range(0, len(header)))
    # print(header)
    # print(header_indexes)
    if not(header_params is None):
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

    for i in range(header_line, len(table)):
        line = clean_split(table[i], delimiter=' ', ex_chars=['', '#'])
        tmp = dict.fromkeys(header)
        for key in header:
            tmp[key] = line[header_dict[key]]
        res.append(tmp)
    return res

def convert_table_to_track(path):
    data = pd.read_table(path, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], delimiter='\s\s\s\s\s\s')
    return data
