import numpy as np

def append_feature(raw, data,flatten=False):
    data = np.array(data)
    if flatten:
        data = data.reshape(-1, 1)
    if raw is None: 
        raw = np.array(data)
    else:
        raw = np.vstack((raw,data))
    return raw