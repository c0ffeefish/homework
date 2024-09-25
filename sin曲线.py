import numpy as np
import matplotlib.pyplot as plt

Eva = 1e-3
count = 67500

x = np.linspace(0, 2 * np.pi, 200) 
y = np.sin(x) 

xMean = np.mean(x)
xStd = np.std(x)
xStand = (x - xMean) / xStd

thetaArray = np.random.randn(6)

def MeanSquareLoss(x, y, thetaArray):
    m = len(y)
    pd = thetaArray[0] + thetaArray[1] * x + thetaArray[2] * (x ** 2) + thetaArray[3] * (x ** 3) + thetaArray[4] * (x ** 4) + thetaArray[5] * (x ** 5)
    
    return (1 / (2 * m)) * np.sum((pd - y) ** 2)

def gradientDescent(x, y, thetaArray, Eva, count):
    m = len(y)
    cost = np.zeros(count)

    for i in range(count):
        pd = thetaArray[0] + thetaArray[1] * x + thetaArray[2] * (x ** 2) + thetaArray[3] * (x ** 3) + thetaArray[4] * (x ** 4) + thetaArray[5] * (x ** 5)

        theta0 = (1 / m) * np.sum(pd - y)
        theta1 = (1 / m) * np.sum((pd - y) * x)
        theta2 = (1 / m) * np.sum((pd - y) * (x ** 2))
        theta3 = (1 / m) * np.sum((pd - y) * (x ** 3))
        theta4 = (1 / m) * np.sum((pd - y) * (x ** 4))
        theta5 = (1 / m) * np.sum((pd - y) * (x ** 5))

        thetaArray[0] -= Eva * theta0
        thetaArray[1] -= Eva * theta1
        thetaArray[2] -= Eva * theta2
        thetaArray[3] -= Eva * theta3
        thetaArray[4] -= Eva * theta4
        thetaArray[5] -= Eva * theta5

        cost[i] = MeanSquareLoss(x, y, thetaArray)
    return thetaArray, cost

thetaArray_gd, cost = gradientDescent(xStand, y, thetaArray, Eva, count)

y_final = thetaArray_gd[0] + thetaArray_gd[1] * xStand + thetaArray_gd[2] * (xStand ** 2) + thetaArray_gd[3] * (xStand ** 3) + thetaArray_gd[4] * (xStand ** 4) + thetaArray_gd[5] * (xStand ** 5)

plt.figure(figsize=(8, 6))
plt.grid(True)

plt.plot(x, y, label='sinx', color='black', linewidth=2)
plt.plot(x, y_final, label='my_sinx', color='red')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')

plt.show()
