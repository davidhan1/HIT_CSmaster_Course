from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy import stats

def gaussCore(u):
    if u:
        return 1/(math.sqrt(2*math.pi))*math.e ** (-0.5 * (u) ** 2)
    return 0

def calc_statistics(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    skew = stats.skew(x)
    kurtosis = stats.kurtosis(x)
    return mu,sigma,skew,kurtosis

new_data = []
with open("./samples.txt", "r") as f:
    data = f.readlines()
for i in data:
    new_data.append(eval(i))
testData = [[new_data[i][0][0], new_data[i][0][1]] for i in range(len(new_data))]
testData = np.array(testData)
testData = testData[0:1500]
#mu, sigma, skew, kurtosis = calc_statistics(testData)
#print('均值、标准差、偏度、峰度：', mu, sigma, skew, kurtosis)
#sample = np.random.randn(200, 2)*0.5
u = [0, 0.7, 1.4]
b = [0.5, 0.7, 0.3]
print(b[2]**2)
X1 = testData[:500, 0]
Y1 = testData[:500, 1]
Z1 = (1.0/(np.pi*2*(b[0]**2)))*np.exp(-0.5*(((X1-u[0])**2)/(b[0]**2)+((Y1-u[0])**2)/(b[0]**2)))
X2= testData[500:1000, 0]
Y2 = testData[500:1000, 1]
Z2 = (1.0/(np.pi*2*(b[1]**2)))*np.exp(-0.5*(((X2-u[1])**2)/(b[1]**2)+((Y2-u[1])**2)/(b[1]**2)))
X3 = testData[1000:1500, 0]
Y3 = testData[1000:1500, 1]
Z3 = (1.0/(np.pi*2*(b[2]**2)))*np.exp(-0.5*(((X3-u[2])**2)/(b[2]**2)+((Y3-u[2])**2)/(b[2]**2)))
X = np.append(X1, np.append(X2, X3))
Y = np.append(Y1, np.append(Y2, Y3))
Z = np.append(Z1, np.append(Z2, Z3))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
plt.show()
