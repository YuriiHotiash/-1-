import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Генерація вибірки з 1000 випадкових значень
np.random.seed(42)
X = np.random.rand(1000, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, 1000)

# Нормалізація значень
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Розділення записів на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Навчання KNN-регресора з різними значеннями K
best_mse = float('inf')
best_k = None
mse_values = []
for k in range(1, 21):  # Перебір різних значень K від 1 до 20
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train, y_train)
    y_pred = knn_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)
    print(f"K = {k}, MSE = {mse}")
    if mse < best_mse:
        best_mse = mse
        best_k = k

print(f"Найкраще значення K: {best_k}, MSE: {best_mse}")

# Візуалізація отриманих рішень
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), mse_values, marker='o', linestyle='--', color='b')
plt.title('MSE vs. K value')
plt.xlabel('K')
plt.ylabel('Mean Squared Error')
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()