from converter import convert_table_to_track
from utils import draw_track

path = 'datasets/test.eep'

track = convert_table_to_track(path)
draw_track(track)