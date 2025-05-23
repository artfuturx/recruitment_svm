# -----------------------------------------------------------------------------
# train_svm_model.py

# Bu dosya, generate_data_faker.py ile üretilmiş gerçekçi aday verilerini kullanarak dört farklı SVM modelini eğitir:
# - Linear kernel
# - RBF  kernel
# - Polynomial kernel 
# - Sigmoid kernel

# Eğitim süreci:
# 1. Veriler 'candidates_data_faker.csv' dosyasından okunur.
# 2. Veriler %80 eğitim, %20 test olarak ayrılır.
# 3. Özellikler StandardScaler ile ölçeklenir.
# 4. Her model eğitilir, başarı oranları yazdırılır ve karar sınırı görselleştirilir.

# Eğitilen modeller ve scaler, tekrar kullanılmak üzere `models` adlı klasör altına .pkl dosyaları olarak kaydedilir:
# - models/scaler.pkl          → Ölçekleyici nesne
# - models/svm_linear.pkl      → Linear SVM modeli
# - models/svm_rbf.pkl         → RBF kernel SVM modeli
# - models/svm_poly.pkl        → Polinom kernel SVM modeli
# - models/svm_sigmoid.pkl     → Sigmoid kernel SVM modeli
#
# Bu sayede modeller her çalıştırmada yeniden eğitilmeden doğrudan FastAPI gibi servislerde kullanılabilir.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os


def plot_decision_boundary_matplotlib(X, y, model, scaler, title):
    plt.figure(figsize=(10, 8))
    
    x_min, x_max = X['years_experience'].min() - 1, X['years_experience'].max() + 1
    y_min, y_max = X['technical_score'].min() - 1, X['technical_score'].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    plt.contour(xx, yy, Z, colors='k', linestyles='-', linewidths=1)
    
    scatter = plt.scatter(X['years_experience'], X['technical_score'], 
                        c=y, s=50, cmap=plt.cm.coolwarm, edgecolors='k')
    
    plt.xlabel('Years of Experience')
    plt.ylabel('Technical Score')
    plt.title(f'Decision Boundary ({title})')
    plt.grid(True, alpha=0.3)
    
    plt.legend(*scatter.legend_elements(), 
             title="Classes", loc="upper right")
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.savefig(f'images/models_images/{title.lower()}.png')
    plt.tight_layout()
    plt.show()

df = pd.read_csv('candidates_data_faker.csv')

X = df[['years_experience', 'technical_score']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler.pkl')

model_configs = {
    'linear': SVC(kernel='linear', probability=True, random_state=42),
    'rbf': SVC(kernel='rbf', probability=True, random_state=42),
    'poly': SVC(kernel='poly', degree=3, probability=True, random_state=42),
    'sigmoid': SVC(kernel='sigmoid', probability=True, random_state=42)
}

for name, model in model_configs.items():
    print(f"\nModel: {name}")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, f'models/svm_{name}.pkl')

    plot_decision_boundary_matplotlib(X, y, model, scaler, title=name.upper())