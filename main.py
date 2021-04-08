from converter import convert_table_to_track, read_table

path = 'datasets/test.eep'

# data = convert_table_to_track('datasets/test.eep')
# # print(data.to_dict())
# print(data.to_dict('split'))

data = read_table(path, 12, header_params=['star_age', 'star_mass', 'he_core_mass'])
print(data)