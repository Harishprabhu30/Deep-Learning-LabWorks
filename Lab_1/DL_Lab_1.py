import numpy as np
import matplotlib.pyplot as plt
import torch

x = np.linspace(-10, 10, 1000)
y = np.sin(2 * x) + np.sin(x)
x = list(x) # convert to list
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = sin(2x) + sin(x)')
plt.show()
