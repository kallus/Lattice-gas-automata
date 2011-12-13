import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('data.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data[:, 0], data[:, 1], 'r-', label='$\mathrm{Box}$ $1$')
ax.plot(data[:, 0], data[:, 2], 'g-', label='$\mathrm{Box}$ $2$')
ax.plot(data[:, 0], data[:, 3], 'b-', label='$\mathrm{Box}$ $3$')

ax.legend()
ax.set_xlabel('$\mathrm{Updates}$')
ax.set_ylabel('$\mathrm{Number}$ $\mathrm{of}$ $\mathrm{particles}$')
ax.set_xlim([0, data[-1, 0]])

plt.show()
