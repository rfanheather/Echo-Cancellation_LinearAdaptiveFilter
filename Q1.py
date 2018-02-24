import numpy as np
from AdaptiveFilter import fir_filter
import matplotlib.pylab as plt

# Import data
training_set = np.loadtxt("corrupted_speech.txt")
noise_set = np.loadtxt("music.txt")

final_erle, final_mse, weight_track, w = fir_filter(training_set, noise_set, 45, 0.00000000000000000000000000000000001)

p1 = plt.plot(weight_track)
#p2 = plt.plot(noise_set)
#plt.axis([0, 10000, -4, 4])
plt.ylabel('Error')
plt.title('Weight tracks across the training data')
#plt.legend((p1[0], p2[0]), ('true function', 'model'), fontsize=6)
plt.show()
