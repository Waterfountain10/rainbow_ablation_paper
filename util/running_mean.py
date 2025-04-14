import numpy as np

def running_mean(x, window_size):
    cumsum = np.cumsum(np.insert(x,0,0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
