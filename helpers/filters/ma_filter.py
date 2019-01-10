import numpy as np


def simple_moving_average(pos_array, winlen):
    """Apply simple moving average filter to a gesture

      Args:
          pos_array:    body keypoint positions to filter
          winlen:       averaging window size (must be odd)
      Returns:
          np.ndarray:   filtered positions
    """

    pos_columns = []
    winlen_oneside = int((winlen - 1) / 2)
    for i in range(len(pos_array[0])):
        line = []
        for j in range(len(pos_array)):
            line.append(pos_array[j][i])
        pos_columns.append(line)

    res_list = []
    for i, joint in enumerate(pos_columns):
        line = []
        for j in range(len(pos_columns[i])):
            start_idx = j - winlen_oneside
            end_idx = j + winlen_oneside + 1
            if start_idx < 0:
                line.append(np.mean(pos_columns[i][:end_idx]))
            elif end_idx > len(pos_columns[i]):
                line.append(np.mean(pos_columns[i][start_idx:]))
            else:
                line.append(np.mean(pos_columns[i][start_idx:end_idx]))
        res_list.append(line)

    res_array = np.array(res_list)

    return res_array.transpose()
