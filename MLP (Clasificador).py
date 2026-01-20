import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from MLP import *

# ======================
# Cargar datos
# ======================
X, Y = load_breast_cancer(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3,
    stratify=Y,
    random_state=0
)

# ======================
# Normalización
# ======================
X_mean_train = X_train.mean(axis=0)
X_std_train  = X_train.std(axis=0)

X_train = (X_train - X_mean_train) / X_std_train
X_test  = (X_test  - X_mean_train) / X_std_train

# ======================
# Modelo
# ======================
np.random.seed(0)
modelo = LayerStack(
    X_train.shape[1],
    [15, 15, 1],
    ["relu", "relu", "sigmoid"],
    0.001,
    "bce"
)

loss = modelo.fit(X_train, Y_train, 10)

# ======================
# Predicciones
# ======================
def predict(modelo, X):
    preds = []
    for x in X:
        x = x.reshape(-1, 1)          # (30,1)
        p = modelo(x)                 # forward
        preds.append(p.item())        # escalar
    return np.array(preds)

pred_train = predict(modelo, X_train)
pred_test  = predict(modelo, X_test)

pred_train = (pred_train > 0.5).astype(int)
pred_test  = (pred_test  > 0.5).astype(int)


# ======================
# Precisión REAL
# ======================
acc_train = np.mean(pred_train == Y_train)
acc_test  = np.mean(pred_test  == Y_test)

print(f"Precisión -> [Entrenamiento] {acc_train:.2f} [Test] {acc_test:.2f}")
