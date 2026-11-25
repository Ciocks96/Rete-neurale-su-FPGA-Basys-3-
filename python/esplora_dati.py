"""Esplora il dataset digits: immagini, pixel e numero di esempi per cifra."""
from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.utils import Bunch

from modello import salva_e_mostra_grafici

# 1. Caricamento
# load_digits() fornisce le immagini in griglia (images), in righe da 64 pixel
# (data) e le etichette delle cifre (target).
# cast serve solo al controllo dei tipi di Pylance; non cambia il risultato.
d = cast(Bunch, load_digits())

print("=== Dimensioni ===")
print("numero di immagini      :", len(d.images))
print("forma di d.images       :", d.images.shape)   # (immagini, righe, colonne)
print("forma di d.data         :", d.data.shape)     # (immagini, pixel)
print("forma di d.target       :", d.target.shape)   # (immagini,)
print("forma di una immagine   :", d.images[0].shape)

# 2. Valori dei pixel
print("\n=== Valori dei pixel ===")
print("tipo dei numeri         :", d.data.dtype)
print("valore minimo           :", d.data.min())
print("valore massimo          :", d.data.max())
valori = np.unique(d.data)                            # valori distinti nel dataset
print("valori diversi presenti :", valori)
print("sono tutti interi?      :", np.all(valori == np.round(valori)))

# Anche la matrice dei pixel permette di riconoscere la cifra.
print("\nPrima immagine (etichetta", d.target[0], ") come numeri:")
print(d.images[0].astype(int))

# 3. Immagini per cifra
print("\n=== Immagini per cifra ===")
conteggi = np.bincount(d.target)                      # conteggi[k]: esempi della cifra k
for cifra, n in enumerate(conteggi):
    print(f"cifra {cifra}: {n} immagini")
print("minimo", conteggi.min(), "- massimo", conteggi.max())

# 4. Un esempio per cifra
fig, assi = plt.subplots(2, 5, figsize=(9, 4))
for cifra in range(10):
    indice = np.where(d.target == cifra)[0][0]        # prima immagine con questa etichetta
    asse = assi[cifra // 5, cifra % 5]                # posizione nella griglia 2x5
    asse.imshow(d.images[indice], cmap="gray_r")      # 0 bianco, 16 nero
    asse.set_title(f"cifra {cifra}")
    asse.axis("off")
fig.suptitle("Una immagine per ogni cifra (8x8 pixel)")
fig.tight_layout()

# 5. Distribuzione dei valori, utile per scegliere quanti bit usare
fig2, asse2 = plt.subplots(figsize=(7, 3.5))
valori, quanti = np.unique(d.data, return_counts=True)
asse2.bar(valori, quanti)
asse2.set_xlabel("valore del pixel")
asse2.set_ylabel("quante volte compare")
asse2.set_title("Distribuzione dei valori dei pixel (tutte le immagini)")
asse2.set_xticks(valori)
fig2.tight_layout()

salva_e_mostra_grafici("esplora_dati")
