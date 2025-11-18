#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ Génération des données ------------------
N = 100
X = np.random.rand(N, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(N, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ Construction modèle ------------------
def build_model(n_hidden_layers, n_neurons, optimizer):
    model = keras.Sequential()
    model.add(layers.Input(shape=(1,)))

    for _ in range(n_hidden_layers):
        model.add(layers.Dense(n_neurons, activation="relu"))

    model.add(layers.Dense(1))
    model.compile(optimizer=optimizer, loss="mse")
    return model

# ------------------ EXACTEMENT 5 CONFIGURATIONS ------------------
configurations = [
    (1, 2, "adam"),
    (2, 4, "sgd"),
    (3, 8, keras.optimizers.Adam(learning_rate=0.01)),
    (1, 8, "adam"),
    (2, 2, keras.optimizers.Adam(learning_rate=0.001)),
]

# ------------------ Dossier principal ------------------
output_dir = "question2"
os.makedirs(output_dir, exist_ok=True)

log_file = open(os.path.join(output_dir, "resultats.txt"), "w")
config_id = 1

# ------------------ Tests ------------------
for hl, nn, opt in configurations:

    # Nom propre optimiseur
    if isinstance(opt, str):
        opt_name = opt
    else:
        opt_name = f"adam_lr_{float(opt.learning_rate)}"

    print(f"[CONFIG {config_id}] HL={hl}, NN={nn}, OPT={opt_name}")

    # Dossier de la configuration
    config_dir = os.path.join(output_dir, f"config_{config_id}")
    os.makedirs(config_dir, exist_ok=True)

    # Construction du modèle
    model = build_model(hl, nn, opt)

    # Entraînement
    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )

    # Évaluation
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    # Prédictions
    y_pred = model.predict(X_test)

    # ---------- courbe perte ----------
    plt.figure()
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title(f"Courbe de perte – Config {config_id}")
    plt.savefig(os.path.join(config_dir, "courbe_perte.png"))
    plt.close()

    # ---------- résumé PNG ----------
    plt.figure(figsize=(8, 6))
    plt.title(f"Résumé – Configuration {config_id}", fontsize=16)

    resume_txt = (
        f"Couches cachées   : {hl}\n"
        f"Neurones/couche   : {nn}\n"
        f"Optimiseur        : {opt_name}\n"
        f"Fonction coût     : MSE\n"
        f"Test Loss final   : {test_loss:.4f}\n"
    )

    plt.text(0.01, 0.5, resume_txt, fontsize=12)
    plt.axis("off")
    plt.savefig(os.path.join(config_dir, "resume.png"))
    plt.close()

    # ---------- graphique prédiction / test ----------
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color='blue', label='Vraies valeurs (Test)')
    plt.scatter(X_test, y_pred, color='red', marker='x', label='Prédictions du modèle')
    
    # Ligne de régression (triée pour être propre)
    sorted_idx = np.argsort(X_test[:, 0])
    plt.plot(X_test[sorted_idx], y_pred[sorted_idx], color='red', linestyle='-', alpha=0.6)

    plt.title(f"Régression RN – Config {config_id} (Test Loss: {test_loss:.4f})")
    plt.xlabel("X_test")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config_dir, "regression_results.png"))
    plt.close()

    print(f"Graphique de régression enregistré : config_{config_id}/regression_results.png")

    # ---------- log ----------
    log_file.write(f"CONFIG {config_id}\n")
    log_file.write(resume_txt + "\n\n")

    config_id += 1

log_file.close()
print("5 configurations testées. Tous les PNG générés dans ./question2/")
