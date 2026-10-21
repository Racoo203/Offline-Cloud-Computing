"""
División privada de MNIST en N_CLIENTS partes estadísticamente equivalentes.

Este archivo y los datos que genera NO forman parte del repositorio (ver
.gitignore): la partición es confidencial. Cada integrante del equipo recibe
únicamente su archivo private/data/client_{k}.npz.

Uso (desde la raíz del proyecto):
    python private/data_setup.py
"""
import os

import numpy as np
import tensorflow as tf
from scipy.stats import chi2_contingency, kruskal
from sklearn.model_selection import StratifiedKFold

N_CLIENTS = 4  # número de integrantes del equipo
SEED = 42
OUT_DIR = "private/data/"

def main():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Estratificar por etiqueta conserva la proporción de cada dígito en cada cliente.
    skf = StratifiedKFold(n_splits=N_CLIENTS, shuffle=True, random_state=SEED)

    class_counts, image_means = [], []
    for k, (_, idx) in enumerate(skf.split(x_train, y_train)):
        x_k, y_k = x_train[idx], y_train[idx]
        np.savez_compressed(os.path.join(OUT_DIR, f"client_{k}.npz"), x=x_k, y=y_k)
        class_counts.append(np.bincount(y_k, minlength=10))
        image_means.append(x_k.reshape(len(x_k), -1).mean(axis=1) / 255.0)

    # Evidencia de equivalencia estadística entre clientes.
    class_counts = np.array(class_counts)
    chi2, p_chi2, dof, _ = chi2_contingency(class_counts)
    h_stat, p_kw = kruskal(*image_means)

    print("Ejemplos por clase (filas = clientes, columnas = dígitos 0-9):")
    for k, row in enumerate(class_counts):
        print(f"  cliente {k}: {row.tolist()}  (n = {row.sum()})")

    print(f"Chi² de homogeneidad de etiquetas: chi2 = {chi2:.4f}, gl = {dof}, p = {p_chi2:.4f}")
    print("Intensidad media por imagen (media ± d.e.):")

    for k, m in enumerate(image_means):
        print(f"  cliente {k}: {m.mean():.4f} ± {m.std():.4f}")

    print(f"Kruskal-Wallis sobre la intensidad media: H = {h_stat:.4f}, p = {p_kw:.4f}")
    print(f"Particiones guardadas en {OUT_DIR}")


if __name__ == "__main__":
    main()
