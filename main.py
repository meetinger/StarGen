from converter import convert_table_to_track, read_table
from utils import draw_track

# path = 'datasets/test.eep'
path = 'datasets/tracks/0001000M.track.eep'

# data = convert_table_to_track('datasets/test.eep')
# # print(data.to_dict())
# print(data.to_dict('split'))

# data = read_table(path, 11, header_params=['star_age', 'star_mass', 'he_core_mass'])
# print(data)



track = convert_table_to_track(path)
# print(track)

draw_track(track)