import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.linspace(0.1,20.1,20)
y = 12+ 10*np.sin(3.14/10*x)+np.random.uniform(-2.8,2.8,len(x))
print(x)
print(y)
plt.scatter(x, y)
csv_array = np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],1)
np.savetxt("09_05.csv",csv_array, delimiter=",", header="x,y", comments='')
plt.show()
