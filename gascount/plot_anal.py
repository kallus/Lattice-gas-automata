import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import numpy as np

t=np.arange(0,30000,1)
k = 1.4032*np.power(10.0,-4.0)

c1=(np.exp(-5*k*t)+4+7*np.exp(-k*t))/12
c2=(1-np.exp(-3*k*t))/3
c3=1/3.0-(np.exp(-5*k*t)+7*np.exp(-k*t)-4*np.exp(-3*k*t))/12

print len(c1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t,c1,'r', label='$\mathrm{Box}$ $1$')
ax.plot(t,c2,'g', label='$\mathrm{Box}$ $1$')
ax.plot(t,c3,'b', label='$\mathrm{Box}$ $1$')

ax.legend()
ax.set_xlabel('$\mathrm{Time}$')
ax.set_ylabel('$\mathrm{Density}$')

plt.show()
