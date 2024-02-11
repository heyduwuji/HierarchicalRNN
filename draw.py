import numpy as np

def func1(t):
    return np.sqrt(20 * np.pi) * np.exp(-5 * t**2)

def func2(t):
    return np.sqrt(20 * np.pi / 11) * np.exp(-5 * t**2 / 11)

# plot func1 and func2 in one figure
import matplotlib.pyplot as plt
t = np.linspace(-1, 1, 100)
plt.plot(t, func1(t), label='func1')
plt.plot(t, func2(t), label='func2')
plt.legend()
plt.show()