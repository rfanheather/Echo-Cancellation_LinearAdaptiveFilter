import numpy as np
import math


def fir_filter(x, target, m, step):
    w = [0] * m
    weight_track = []

    for k in range(m, x.size):
        x_k = x[k - m: k][:: -1]
        y = np.dot(w, x_k)  # Signal obtained after the filter
        err = abs(target[k] - y)

        # Update w with LMS
        w_new = w + 2 * step * err * x_k
        w = w_new
        weight_track.append(w)

    # Compute ERLE
    mse = 0
    erle = 0

    for k in range(m, x.size):
        x_k = x[k - m: k][:: -1]
        y_predict = np.dot(w, x_k)

        err = abs(x[k]) / abs(target[k] - y_predict)
        new_erle = erle + err
        erle = new_erle

        #err = abs(target[k] - y_predict)
        #se = err**2
        #new_mse = mse + se
        #mse = new_mse

    final_mse = mse / (x.size - m)

    mean_erle = erle / (x.size - m)
    erle = abs(10 * math.log10(mean_erle))

    return erle, final_mse, weight_track, w
