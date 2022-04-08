import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Nama kolom dari dataset iris
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Load dataset dengan pandas
dataset = pd.read_csv('iris.data', names=names)
# tampilkan format data
dataset.head()

# pisahkan independen variable dengan dependen variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# lakukan spliting data train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# load library KNN dengan nilai kedekatan 5
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# prdiksi data test
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

error = []
# lakukan pencarian nilai tetangga terdekat untuk prediksi terbaik
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# tampilkan plot nilai prediksi terbaik
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
