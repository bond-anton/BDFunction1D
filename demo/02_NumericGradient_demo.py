from BDFunction1D.Standard import Sin
from BDFunction1D.Interpolation import InterpolateFunction
from BDFunction1D.Differentiation import NumericGradient

import numpy as np
from matplotlib import pyplot as plt


x = np.linspace(0.0, 2*np.pi, num=15, endpoint=True)
y = np.sin(x)
err = np.ones_like(x) * 0.1

f1 = InterpolateFunction(x, y, err)
f2 = Sin()

f1_grad = NumericGradient(f1)
f2_grad = NumericGradient(f2)

x1 = np.linspace(0.0, 2*np.pi, num=500, endpoint=True)
y1 = np.asarray(f1.evaluate(x1))
y2 = np.asarray(f2.evaluate(x1))
err1 = f1.error(x1)
err2 = f2.error(x1)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(x, y, 'o-r')

ax1.plot(x1, y1, '-b')
ax1.plot(x1, y1 + err2, '-b')
ax1.plot(x1, y1 - err2, '-b')

ax1.plot(x1, y2, '-g')
ax1.plot(x1, y2 + err2, '-g')
ax1.plot(x1, y2 - err2, '-g')

ax2.plot(x, err, 'o-r')
ax2.plot(x1, err1, '-b')
ax2.plot(x1, err2, '-g')

y1 = np.asarray(f1_grad.evaluate(x1))
y2 = np.asarray(f2_grad.evaluate(x1))
err1 = f1_grad.error(x1)
err2 = f2_grad.error(x1)

ax3.plot(x1, y1, '-b')
ax3.plot(x1, y1 + err2, '-b')
ax3.plot(x1, y1 - err2, '-b')

ax3.plot(x1, y2, '-g')
ax3.plot(x1, y2 + err2, '-g')
ax3.plot(x1, y2 - err2, '-g')

ax4.plot(x, err, 'o-r')
ax4.plot(x1, err1, '-b')
ax4.plot(x1, err2, '-g')

plt.show()
