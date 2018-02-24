import numpy as np
from AdaptiveFilter import gamma_filter
import matplotlib.pylab as plt
from scipy.io.wavfile import write

# Import data
desire = np.tile(np.loadtxt("corrupted_speech.txt"), 1)
noise = np.tile(np.loadtxt("music.txt"), 1)
fs = 22000

ERLE = {}
for m in range(5, 105, 5):
    J, Wtrack, erle = gamma_filter(noise, desire, m, 0.2, 0.01)
    ERLE[m] = erle

print(ERLE)
print(max(ERLE, key=ERLE.get))
print(ERLE[max(ERLE, key=ERLE.get)])

ERLE_lists = sorted(ERLE.items())
x, y = zip(*ERLE_lists)
plt.plot(x, y)
plt.xlabel('Number of Taps')
plt.ylabel('ERLE(dB)')
plt.title('ERLE of the Gamma filter as a function of the number of taps')
plt.show()


m_best = max(ERLE, key=ERLE.get)
J2, Wtrack2, erle2 = gamma_filter(noise, desire, m_best, 0.2, 0.01)

desire = np.loadtxt("corrupted_speech.txt")
noise = np.loadtxt("music.txt")

y_predict = np.zeros(noise.size - 1)
x_input = np.zeros(shape=(m_best, noise.size))
x_input[0, :] = noise

for i in range(1, noise.size):
    x_i_0 = x_input[:, i - 1]
    x_i = x_input[:, i]

    for k in range(1, m_best):
        x_i[k] = (1 - 0.2) * x_i_0[k] + 0.2 * x_i_0[k - 1]

    # Update x
    x_input[:, i] = x_i

    y = np.dot(Wtrack2[-1], x_i)  # Signal obtained after the filter
    y_predict[i - 1] = desire[i] - y

scaled = np.int16(y_predict/np.max(np.abs(y_predict)) * 32767)
write('gamma.wav', fs, scaled)
