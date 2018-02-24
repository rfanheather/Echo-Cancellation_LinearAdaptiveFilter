import numpy as np
import math


def fir_filter(x, target, m, step):
    w = [0] * m

    # Cost Function
    J = []
    Wtrack = np.zeros(shape=(x.size - m, m))

    for k in range(m, x.size):
        x_k = x[k - m: k][:: -1]
        y = np.dot(w, x_k)  # Signal obtained after the filter
        err = target[k] - y

        # Cost Function
        J.append(err**2)

        # Update w with normalized LMS
        w_new = w + 2 * step * err * x_k / np.dot(x_k.transpose(), x_k)
        w = w_new
        Wtrack[k - m, :] = w

    # Compute ERLE
    err_sum = 0
    E_d_square = sum(i * i for i in target) / target.size

    for k in range(m, x.size):
        x_k = x[k - m: k][:: -1]
        y_predict = np.dot(Wtrack[x.size - m - 1, :].transpose(), x_k)

        err = target[k] - y_predict
        new_err_sum = err_sum + err**2
        err_sum = new_err_sum

    #E_err_sqr = err_sum / (x.size - m)
    E_err_sqr = sum(J) / (x.size - m)

    erle = 10 * math.log10(E_d_square / E_err_sqr)

    return J, Wtrack, erle


def gamma_filter(x, target, m, mu, step):

    print(m)

    w = [0] * m
    x_input = np.zeros(shape=(m, x.size))
    x_input[0, :] = x
    x_test = x

    # Cost Function
    J = []

    Wtrack = np.zeros(shape=(x.size, m))

    for i in range(1, x.size):
        x_i_0 = x_input[:, i - 1]
        x_i = x_input[:, i]

        for k in range(1, m):
            x_i[k] = (1 - mu) * x_i_0[k] + mu * x_i_0[k - 1]

        # Update x
        x_input[:, i] = x_i

        y = np.dot(w, x_i)  # Signal obtained after the filter
        err = target[i] - y

        # Cost Function
        J.append(err**2)

        # Update w with normalized LMS
        w_new = w + 2 * step * err * x_i
        #print(w_new)
        w = w_new
        Wtrack[i] = w

    # Compute ERLE
    err_sum = 0
    E_d_square = sum(i * i for i in target) / target.size

    for i in range(1, x.size):
        x_i = x_input[:, i]

        y = np.dot(w, x_i)  # Signal obtained after the filter

        err = target[i] - y
        new_err_sum = err_sum + err ** 2
        err_sum = new_err_sum

    #E_err_sqr = err_sum / (x.size - 1)

    E_err_sqr = sum(J) / (x.size - 1)

    erle = 10 * math.log10(E_d_square / E_err_sqr)

    return J, Wtrack, erle
