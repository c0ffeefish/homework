import matplotlib.pyplot as plt
years = list(range(2010, 2024, 1))
price = [10878, 14654, 15467, 18293, 17450, 19997, 21553, 28508, 33289, 31951, 31732, 35655, 38514, 36313]

n = 14
x = sum(years) / n
y = sum(price) / n

k = (sum(years[i] * price[i] for i in range(n)) - n * x * y) / (sum(years[i] ** 2 for i in range(n)) - n * x * x)
b = y - k * x
print(k, b)

plt.scatter(years, price)
plt.plot(years, [k * years[i] + b for i in range(n)], color = (0.7, 0.5, 0.5))
plt.xlabel('year')
plt.ylabel('price')

plt.show()