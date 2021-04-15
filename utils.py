
import matplotlib.pyplot as plt

from converter import get_column_from_table_dict


def draw_track(data):
    track = data['track']
    x = get_column_from_table_dict(track, 'log_Teff')
    y = get_column_from_table_dict(track, 'log_L')
    plt.plot(x, y)
    plt.xlabel('log_Teff')
    plt.ylabel('log_L')
    plt.gca().invert_xaxis()
    plt.show()
