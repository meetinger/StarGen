import pandas as pd


def read_table(path, header_line, delimiter=" ", header_params=None):
    # with open(path, 'r') as table:
    #     for _ in range(header):
    #         table.next()
    table = open(path, 'r').readlines()
    # print(table)
    header = table[header_line - 1].split(delimiter)
    header = [s.strip() for s in header if (s != '') and (s != '#')]
    header_indexes = list(range(0, len(header)))
    print(header)
    # if not(header_params is None):
    #     header = header_params
    #
    # for i in range(header_line, len(table)):
    #     print(table[i])


def convert_table_to_track(path):
    data = pd.read_table(path, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], delimiter='\s\s\s\s\s\s')
    return data
