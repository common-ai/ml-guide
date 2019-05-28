from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 

def thermometer(x, start, end, step_size=1):
    thresholds = np.arange(start, end, step_size)
    thermo = (x > thresholds).astype(float)
    thermo[np.arange(len(x)), ((x - start) // step_size).astype(int).reshape(len(x))] = np.fmod(x, step_size).reshape(len(x))
    return thermo

X = np.random.uniform(0, 10, size=(50, 1))
X = np.sort(X, axis=0)
X_val = np.random.uniform(0, 10, size=(50, 1))
X_val = np.sort(X_val, axis=0)

X_thermo = thermometer(X, 0, 10, step_size=1)
X_thermo_val = thermometer(X_val, 0, 10, step_size=1)

y = np.square(X) + 5.0
y_val = np.square(X_val) + 5.0

lr = LinearRegression()
lr.fit(X, y)
print(lr.score(X_val, y_val))

lr2 = LinearRegression()
lr2.fit(X_thermo, y)
print(lr2.score(X_thermo_val, y_val))

plt.plot(X_val, lr.predict(X_val), label='Raw Feature')
plt.plot(X_val, lr2.predict(X_thermo_val),label='Thermometer Encoding')
plt.plot(X_val, y_val,'ro', label='True Value')
plt.legend()
plt.savefig('../figures/thermo.pdf')