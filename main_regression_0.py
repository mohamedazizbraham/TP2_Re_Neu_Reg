#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import des bibliothèques
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------- Génération des données ----------------------
N = 100
X = np.random.rand(N, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(N, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------- Modèle à compléter --------------------------
model = keras.Sequential([
    layers.Input(shape=(1,)),          # Couche d'entrée
    layers.Dense(2, activation="relu"),# Couche cachée : 2 neurones
    layers.Dense(1)                    # Couche de sortie (linéaire)
])

model.compile(optimizer='adam', loss='mse')  # MSE pour la régression

# ---------------------- Entraînement -------------------------------
Nb_epochs = 80
Batchsize = 32

history = model.fit(
    X_train, y_train,
    epochs=Nb_epochs,
    batch_size=Batchsize,
    validation_data=(X_test, y_test)
)

# ---------------------- Évaluation -------------------------------
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")

# ---------------------- Création d’un dossier pour Question 1 ----------------------
output_dir = "question1"
os.makedirs(output_dir, exist_ok=True)

# Sauvegarde du modèle
model.save(os.path.join(output_dir, "modele_question1.keras"))

# Sauvegarde de l’historique d'entraînement
np.save(os.path.join(output_dir, "history.npy"), history.history)

# Sauvegarde du graphique
plt.figure()
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.legend()
plt.title("Courbe de perte - Question 1")
plt.savefig(os.path.join(output_dir, "courbe_perte.png"))
plt.close()

print("Tous les résultats de la question 1 ont été enregistrés dans : ./question1")
