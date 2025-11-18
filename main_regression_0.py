#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import os
import tensorflow as tf
from tensorflow import keras

# ------------------ Nouveau jeu de données ------------------
N = 100
X = np.random.rand(N, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(N, 1)

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# ------------------ Dossier de sortie ------------------
output_dir = "question3"
os.makedirs(output_dir, exist_ok=True)

# ------------------ Régression avec le meilleur RN ------------------
# Ici, on recrée un modèle identique au meilleur (à remplacer selon Q2)
# Exemple : 2 couches cachées, 4 neurones, optimizer=Adam 0.01
best_model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(2, activation="relu"),
    keras.layers.Dense(1)
])
best_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# Entraînement sur le nouveau jeu de données
best_model.fit(X_train, y_train, epochs=80, batch_size=32, verbose=0)

# Prédiction
y_pred_rn = best_model.predict(X_test)
mse_rn = mean_squared_error(y_test, y_pred_rn)
print("MSE RN :", mse_rn)

# ---------- PNG Régression RN ----------
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label='Vraies valeurs (Test)')
plt.scatter(X_test, y_pred_rn, color='red', marker='x', label='Prédictions RN')
plt.plot(X_test[np.argsort(X_test[:,0])], y_pred_rn[np.argsort(X_test[:,0])], color='red', alpha=0.6)
plt.title(f"Régression RN – MSE: {mse_rn:.4f}")
plt.xlabel("X_test")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "regression_rn.png"))
plt.close()

# ------------------ Régression Linéaire ------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
print("MSE Linéaire :", mse_lin)

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label='Vraies valeurs (Test)')
plt.scatter(X_test, y_pred_lin, color='green', marker='x', label='Prédictions Linéaire')
plt.plot(X_test[np.argsort(X_test[:,0])], y_pred_lin[np.argsort(X_test[:,0])], color='green', alpha=0.6)
plt.title(f"Régression Linéaire – MSE: {mse_lin:.4f}")
plt.xlabel("X_test")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "regression_lineaire.png"))
plt.close()

# ------------------ Régression Polynomiale ------------------
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
print("MSE Polynomial (deg3) :", mse_poly)

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label='Vraies valeurs (Test)')
plt.scatter(X_test, y_pred_poly, color='orange', marker='x', label='Prédictions Poly3')
plt.plot(X_test[np.argsort(X_test[:,0])], y_pred_poly[np.argsort(X_test[:,0])], color='orange', alpha=0.6)
plt.title(f"Régression Polynomiale (deg3) – MSE: {mse_poly:.4f}")
plt.xlabel("X_test")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "regression_polynomial.png"))
plt.close()

# ------------------ Résumé comparatif ------------------
plt.figure(figsize=(8,5))
plt.axis('off')
txt = (
    f"Comparaison des modèles – Question 3\n\n"
    f"Réseau de neurones      : MSE = {mse_rn:.4f}\n"
    f"Régression Linéaire      : MSE = {mse_lin:.4f}\n"
    f"Régression Polynomiale   : MSE = {mse_poly:.4f}\n"
)
plt.text(0.01, 0.5, txt, fontsize=12)
plt.savefig(os.path.join(output_dir, "resume_comparatif.png"))
plt.close()

print("Question 3 terminée. Tous les PNG sont dans ./question3/")
