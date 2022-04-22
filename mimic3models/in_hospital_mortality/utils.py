from __future__ import absolute_import
from __future__ import print_function

from mimic3models import common_utils
import numpy as np
import os


def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    print('Number of examples: ', N)
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    # print('ret',np.shape(ret))
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    print('discretized data example', data[0][0],'\nshape:',np.shape(data))
    
    if normalizer is not None:
        datsadasa = [normalizer.transform(X) for X in data]
    
    print('normalized data example', data[0][0],'\nshape:',np.shape(data))
    whole_data = (np.array(data), np.array(labels))
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
