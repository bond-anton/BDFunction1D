from BDFunction1D.Interpolation import InterpolateFunction

import numpy as np
from matplotlib import pyplot as plt


x = np.linspace(0.0, 2*np.pi, num=10, endpoint=True)
y = np.sin(x)
err = np.ones_like(x) * 0.1
# err = np.arange(10) * 0.1

f = InterpolateFunction(x, y, err)

x1 = np.linspace(0.0, 2*np.pi, num=1000, endpoint=True)
y1 = np.asarray(f.evaluate(x1))
err1 = f.error(x1)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(x, y, 'o-r')
ax1.plot(x1, y1, '-')
ax1.plot(x1, y1+err1, '-b')
ax1.plot(x1, y1-err1, '-b')
ax2.plot(x, err, 'o-r')
ax2.plot(x1, err1, '-')
plt.show()
