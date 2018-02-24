import numpy as np
from AdaptiveFilter import fir_filter
import matplotlib.pylab as plt
from scipy.io.wavfile import write

# Import data
desire = np.tile(np.loadtxt("corrupted_speech.txt"), 1)
noise = np.tile(np.loadtxt("music.txt"), 1)
fs = 22000

ERLE = {}
for m in range(5, 1000, 100):
    J, Wtrack, erle = fir_filter(noise, desire, m, 0.0001)
    ERLE[m] = erle

print(ERLE)
print(max(ERLE, key=ERLE.get))
print(ERLE[max(ERLE, key=ERLE.get)])

ERLE_lists = sorted(ERLE.items())
x, y = zip(*ERLE_lists)
plt.plot(x, y)
plt.xlabel('Number of Taps')
plt.ylabel('ERLE(dB)')
plt.title('ERLE of the FIR filter as a function of the number of taps')
plt.show()


m_best = max(ERLE, key=ERLE.get)
J2, Wtrack2, erle2 = fir_filter(noise, desire, m_best, 0.01)

desire = np.loadtxt("corrupted_speech.txt")
noise = np.loadtxt("music.txt")

y_predict = np.zeros(noise.size - m_best)
for k in range(m_best, noise.size):
    x_k = noise[k - m_best: k][:: -1]
    y_predict[k - m_best] = desire[k] - np.dot(Wtrack2[-1], x_k)

scaled = np.int16(y_predict/np.max(np.abs(y_predict)) * 32767)
#write('fir.wav', fs, scaled)
