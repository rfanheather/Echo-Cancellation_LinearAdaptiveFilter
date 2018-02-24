import numpy as np
from AdaptiveFilter import gamma_filter
import matplotlib.pylab as plt
from scipy.io.wavfile import write

# Import data
desire = np.loadtxt("corrupted_speech.txt")
noise = np.loadtxt("music.txt")
fs = 22000

m = 40

J, Wtrack, erle = gamma_filter(noise, desire, m, 0.2, 0.0001)

y_predict = np.zeros(noise.size - m)
for k in range(m, noise.size):
    x_k = noise[k - m: k][:: -1]
    y_predict[k - m] = desire[k] - np.dot(Wtrack[-1], x_k)

scaled = np.int16(y_predict/np.max(np.abs(y_predict)) * 32767)
write('test2.wav', fs, scaled)

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
