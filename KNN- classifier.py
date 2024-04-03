import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Завантаження даних iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Перемішуємо записи
np.random.seed(42)  # Для відтворюваності результатів
shuffle_index = np.random.permutation(len(X))
X_shuffled, y_shuffled = X[shuffle_index], y[shuffle_index]

# Нормалізація параметрів
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_shuffled)

# Розділення на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_shuffled, test_size=0.2, random_state=42)

# Навчання KNN-класифікатора з різними значеннями K
k_values = [3, 5, 7, 9]  # Значення K, які будемо перевіряти
best_accuracy = 0
best_k = None

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"K={k}, Точність: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"Найкраще значення K: {best_k}, Точність: {best_accuracy}")