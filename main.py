from converter import convert_table_to_track, read_table, create_dataset
from utils import draw_track

path = 'datasets/test.eep'
# path = 'datasets/tracks/0001000M.track.eep'

# data = convert_table_to_track('datasets/test.eep')
# # print(data.to_dict())
# print(data.to_dict('split'))

data = convert_table_to_track(path)
# print(data)
print(create_dataset(data))



# track = convert_table_to_track(path)
# print(track)

# draw_track(track)