from converter import convert_table_to_track
from utils import draw_track

path = 'datasets/tracks/0010000M.track.eep'

track = convert_table_to_track(path)
draw_track(track)