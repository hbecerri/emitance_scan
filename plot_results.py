import matplotlib.pyplot as plt
import numpy as np
from statistics import *
 
X, Y = np.loadtxt('readme.txt', delimiter=',', unpack=True)
plt.hist(X, bins=27, range=(-8, 1))
#plt.hist(Y, bins=27, range=(0, 1))
plt.title('slope - m * xavg-SBIL + b')
plt.xlabel('m')
plt.ylabel('Events')
plt.text(-7,17.5, 'median = %f'%(median(Y)))
plt.text(-7,15,  'stdev = %f'%(stdev(Y)))
plt.savefig('slope.pdf')

