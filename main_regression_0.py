#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importer les bibliothèques nécessaires :
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration pour l'enregistrement des résultats ---
QUESTION_NUMBER = "1"
OUTPUT_DIR = os.path.join(os.getcwd(), QUESTION_NUMBER)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------------------------


# ----------------------------------------------------------------- #
# ---------------  Générer quelques données  --------------------- #
# ----------------------------------------------------------------- #
N = 100
X = np.random.rand(N, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(N, 1) # Relation linéaire y = 2x + 1 + bruit

# Séparer les données en ensembles d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Afficher les données générées (tel que demandé dans le TP)
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Données générées')
plt.title("Données de Régression Générées")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "data_plot.png"))
plt.close() # Fermer la figure pour ne pas afficher immédiatement
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #


# ----------------------------------------------------------------- #
# ------------------------    le modèle  ------------------------- #
# ----------------------------------------------------------------- #

# La couche de sortie doit avoir 1 neurone sans fonction d'activation
# (ou activation 'linear') pour la régression.
model = keras.Sequential([
    layers.Input(shape=(1,)),                                      # Couche d'entrée: 1 seule caractéristique
    layers.Dense(2, activation='relu'),                            # Couche cachée: 2 neurones, activation ReLU
    layers.Dense(1, activation='linear')                           # Couche de sortie: 1 neurone, activation linéaire (pour la régression)
])

# Fonction de coût adaptée à la régression : Mean Squared Error (MSE)
model.compile(optimizer='adam', loss='mse')

model.summary()
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #


# ----------------------------------------------------------------- #
# -------------------    Entrainement du modèle  ----------------- #
# ----------------------------------------------------------------- #
Nb_epochs = 80
Batchsize = 32
print("\n--- Entraînement du modèle ---")
history = model.fit(X_train, y_train, 
                    epochs=Nb_epochs, 
                    batch_size=Batchsize, 
                    validation_data=(X_test, y_test),
                    verbose=0) # verbose=0 pour un affichage plus propre dans le terminal
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# -------------------     Evaluation du modèle   ----------------- #
# ----------------------------------------------------------------- #
print("\n--- Évaluation du modèle ---")
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {test_loss:.4f}")
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #


# ----------------------------------------------------------------- #
# -------------------        Tester modèle       ----------------- #
# ----------------------------------------------------------------- #
# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# Afficher les résultats
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Vraies valeurs (Test)')
plt.scatter(X_test, y_pred, color='red', marker='x', label='Prédictions du modèle')
plt.plot(X_test, y_pred, color='red', linestyle='-', alpha=0.5) # Ligne de régression (approximative)
plt.title(f"Régression Linéaire par RN (Test Loss: {test_loss:.4f})")
plt.xlabel("X_test")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "regression_results.png"))
print(f"Graphique des résultats de régression enregistré dans le répertoire '{QUESTION_NUMBER}/'")
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #


# ----------------------------------------------------------------- #
# ------------------- Régression polynomiale     ----------------- #
# ----------------------------------------------------------------- #
# Ces lignes étaient dans votre code initial mais ne sont pas nécessaires pour la Q1.
# Elles ont été laissées telles quelles pour l'intégrité du code initial.
degre_polynome = 3
[ll,cc] = X_train.shape
xx= np.reshape(X_train,(ll))
yy = np.reshape(y_train,(ll))
model_poly = np.poly1d(np.polyfit(xx, yy, degre_polynome))
# Pour pouvoir exécuter cette ligne, nous définissons une `new_data`
new_data = np.linspace(0, 1, 100) 
prediction_poly = model_poly(new_data.reshape(-1, 1))
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #

# Afficher l'évolution de la perte d'entraînement (Optionnel mais instructif)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Loss (Training)')
plt.plot(history.history['val_loss'], label='Loss (Validation/Test)')
plt.title('Évolution de la fonction de coût')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_history.png"))
print(f"Historique de la perte enregistré dans le répertoire '{QUESTION_NUMBER}/'")
plt.show() # Afficher toutes les figures à la fin