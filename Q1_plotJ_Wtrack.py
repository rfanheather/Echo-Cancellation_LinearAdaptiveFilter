import numpy as np
from AdaptiveFilter import fir_filter
import matplotlib.pylab as plt
from scipy.io.wavfile import write

# Import data
desire = np.tile(np.loadtxt("corrupted_speech.txt"), 1)
noise = np.tile(np.loadtxt("music.txt"), 1)
fs = 22000

m = 380

J, Wtrack, erle = fir_filter(noise, desire, m, 0.001)

y_predict = np.zeros(noise.size - m)
for k in range(m, noise.size):
    x_k = noise[k - m: k][:: -1]
    y_predict[k - m] = desire[k] - np.dot(Wtrack[-1], x_k)

scaled = np.int16(y_predict/np.max(np.abs(y_predict)) * 32767)
write('fir2.wav', fs, scaled)

plt.figure(1)
plt.title('Weight tracks and mean square error across the training data')

plt.subplot(211)
plt.plot(J)
plt.xlabel('Sample')
plt.ylabel('Mean Square Error')

plt.subplot(212)
for i in range(0, m):
    plt.plot(Wtrack[:, i])
plt.xlabel('Sample')
plt.ylabel('Weights')
plt.show()
