"""Funzioni e costanti condivise dagli script del progetto.

Include il caricamento dei dati, l'addestramento e la quantizzazione. Il VHDL
verrà confrontato con predici_intero(), che definisce il calcolo a interi.
Eseguendo questo file si controlla la divisione dei dati.
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import Bunch

# Costanti condivise dagli script
PIXEL_MAX = 15          # pixel a 4 bit; il valore 16 viene saturato a 15
SCALA_INGRESSO = 16     # in addestramento i pixel si dividono per 16 (= shift di 4 bit)
SEME = 0                # mantiene fissa la divisione tra addestramento e test
FRAZIONE_TEST = 0.25    # un quarto dei dati è riservato al test
N_NASCOSTI = 16         # dimensione dello strato nascosto
SEME_RETE = 0           # inizializzazione della rete usata nel progetto
BIT_PESI = 8            # pesi con segno: da -127 a 127
BIT_H = 8               # uscite nascoste senza segno: da 0 a 255

CARTELLA_GRAFICI = Path(__file__).resolve().parent / "grafici"
MOSTRA_GRAFICI = True   # False per salvare i grafici senza aprire finestre


# Grafici
def salva_e_mostra_grafici(nome: str) -> None:
    """
    Salva le figure aperte in grafici/<nome>.png e, se richiesto, le mostra.
    plt.show() tiene lo script in attesa finché non si chiudono le finestre.
    """
    import matplotlib.pyplot as plt

    CARTELLA_GRAFICI.mkdir(exist_ok=True)
    figure = plt.get_fignums()
    for k, numero in enumerate(figure, 1):
        suffisso = "" if len(figure) == 1 else f"_{k}"
        file = CARTELLA_GRAFICI / f"{nome}{suffisso}.png"
        plt.figure(numero).savefig(file, dpi=120)
        print(f"grafico salvato: grafici/{file.name}")

    if MOSTRA_GRAFICI and figure:
        print("\n>>> Finestre dei grafici aperte (forse dietro VS Code: icona di Python nel Dock).")
        print(">>> Lo script termina quando le chiudi.")
        plt.show()


# Passo 2: caricamento e divisione dei dati
def carica_dati(pixel_max: int | None = PIXEL_MAX
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica digits, limita i pixel a pixel_max e divide i dati in addestramento
    e test mantenendo la stessa proporzione di cifre in entrambi i gruppi.

    pixel_max è il valore massimo dei pixel; con None non viene applicata
    la saturazione (pixel originali 0..16).

    Restituisce X_train, X_test, y_train e y_test. Le X contengono le immagini
    come righe di 64 pixel; le y contengono le cifre corrispondenti.
    """
    d = cast(Bunch, load_digits())

    # Ogni immagine diventa una riga di 64 pixel, convertiti da float a interi.
    X = d.data.astype(np.int64)
    # Limita i pixel al valore massimo scelto (nel dataset 16 diventa 15).
    if pixel_max is not None:
        X = np.minimum(X, pixel_max)
    y = d.target

    # Mantiene la proporzione delle cifre nei due gruppi.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=FRAZIONE_TEST, random_state=SEME, stratify=y)
    # Specifica a Pylance i tipi restituiti da train_test_split.
    return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)


# Passi 4-5: rete in virgola mobile
def addestra_rete(X: np.ndarray, y: np.ndarray, n_nascosti: int = N_NASCOSTI,
                  seme: int = SEME_RETE) -> MLPClassifier:
    """
    Addestra una rete 64 -> n_nascosti (ReLU) -> 10.
    Divide gli ingressi per SCALA_INGRESSO (16) prima dell'addestramento.
    Essendo una potenza di 2, in hardware la divisione diventa parte della
    scala dei pesi (quantizza.py). I valori predefiniti corrispondono alla
    rete scelta per il progetto.
    """
    rete = MLPClassifier(hidden_layer_sizes=(n_nascosti,),   # un solo strato nascosto
                         activation="relu",
                         max_iter=3000,                      # massimo numero di iterazioni
                         random_state=seme)                  # inizializzazione ripetibile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        rete.fit(X / SCALA_INGRESSO, y)
    return rete


# Passo 7: quantizzazione e modello a interi
@dataclass
class ReteIntera:
    """Pesi, bias e parametri della rete quantizzata."""
    W1: np.ndarray      # 64 x 16, interi con segno (BIT_PESI bit)
    b1: np.ndarray      # 16 bias interi (alla scala dei prodotti peso x pixel)
    W2: np.ndarray      # 16 x 10, interi con segno (BIT_PESI bit)
    b2: np.ndarray      # 10 bias interi (alla scala dei prodotti peso x uscita nascosta)
    shift: int          # riquantizzazione tra gli strati: h = acc1 >> shift
    bit_h: int          # bit delle uscite nascoste (saturazione a 2^bit_h - 1)
    scala1: float       # peso intero strato 1 = round(peso reale / 16 x scala1)
    scala2: float       # peso intero strato 2 = round(peso reale x scala2)
    # Con scale a potenza di 2, scala = 2^n: n è il numero di bit frazionari
    # (virgola fissa). Esempio: scala2 = 2^6 -> pesi in formato Q2.6.


def quantizza_simmetrico(W: np.ndarray, bit: int, potenza_di_2: bool = True
                         ) -> tuple[np.ndarray, float]:
    """Quantizza W usando lo stesso intervallo positivo e negativo; lo zero resta zero.

    Scala libera: il valore massimo in modulo diventa 2^(bit-1) - 1 (127 con 8 bit).
    Potenza di 2 (virgola fissa, come nel progetto di riferimento): si usa la
    potenza di 2 più grande che non supera la scala libera, quindi il massimo
    finisce tra 64 e 127.
    """
    massimo = 2 ** (bit - 1) - 1
    scala = massimo / float(np.abs(W).max())
    if potenza_di_2:
        scala = 2.0 ** np.floor(np.log2(scala))
    Wq = np.clip(np.round(W * scala), -massimo, massimo).astype(np.int64)
    return Wq, scala


def predici_intero(q: ReteIntera, X: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Esegue la rete quantizzata e restituisce cifre, h, acc1 e acc2.

    Tra i due strati applica ReLU, shift e saturazione. In caso di parità
    tra punteggi, argmax sceglie la cifra con indice più basso.
    """
    h_max = 2 ** q.bit_h - 1
    acc1 = X.astype(np.int64) @ q.W1 + q.b1
    h = np.minimum(h_max, np.maximum(acc1, 0) >> q.shift)
    acc2 = h @ q.W2 + q.b2
    return acc2.argmax(axis=1), h, acc1, acc2


def quantizza_rete(rete: MLPClassifier, X_cal: np.ndarray, y_cal: np.ndarray,
                   bit_pesi: int = BIT_PESI, bit_h: int = BIT_H,
                   potenza_di_2: bool = True
                   ) -> tuple[ReteIntera, list[tuple[int, float, float]]]:
    """Converte la rete e sceglie lo shift usando X_cal e y_cal.

    La calibrazione va fatta sui dati di addestramento; il test serve solo
    a misurare il risultato. Restituisce la rete intera e le prove
    (shift, accuratezza %, uscite sature %).

    W1 viene diviso per SCALA_INGRESSO prima di essere scalato, perché i pixel
    in ingresso sono interi. Con scale a potenza di 2 anche questa divisione
    è solo uno spostamento della virgola. Per mantenere le scale degli accumulatori:
    b1 usa scala1; dopo lo shift, b2 usa scala2 * scala1 / 2^shift.
    La scala finale è positiva e comune a tutti i punteggi, quindi non cambia
    la cifra scelta.
    """
    W1_reale = rete.coefs_[0] / SCALA_INGRESSO       # divisione per 16 assorbita nei pesi
    b1_reale, b2_reale = rete.intercepts_
    W2_reale = rete.coefs_[1]

    W1q, scala1 = quantizza_simmetrico(W1_reale, bit_pesi, potenza_di_2)
    b1q = np.round(b1_reale * scala1).astype(np.int64)
    W2q, scala2 = quantizza_simmetrico(W2_reale, bit_pesi, potenza_di_2)

    prove: list[tuple[int, float, float]] = []
    migliore: tuple[float, float, int] | None = None
    for shift in range(0, 21):
        b2q = np.round(b2_reale * scala2 * scala1 / 2 ** shift).astype(np.int64)
        q = ReteIntera(W1q, b1q, W2q, b2q, shift, bit_h, scala1, scala2)
        cifre, h, _, _ = predici_intero(q, X_cal)
        acc = float(np.mean(cifre == y_cal) * 100)
        sature = float(np.mean(h == 2 ** bit_h - 1) * 100)
        prove.append((shift, acc, sature))
        # A parità di accuratezza preferisce meno saturazioni, poi lo shift minore.
        chiave = (acc, -sature, -shift)
        if migliore is None or chiave > migliore:
            migliore = chiave
    assert migliore is not None
    shift = -migliore[2]
    b2q = np.round(b2_reale * scala2 * scala1 / 2 ** shift).astype(np.int64)
    return ReteIntera(W1q, b1q, W2q, b2q, shift, bit_h, scala1, scala2), prove


# Passo 9: quanti bit servono agli accumulatori
def bit_con_segno(minimo: int, massimo: int) -> int:
    """Numero di bit in complemento a 2 per rappresentare tutti i valori tra minimo e massimo.

    Con n bit con segno si va da -2^(n-1) a 2^(n-1) - 1.
    """
    n = 1
    while not (-(2 ** (n - 1)) <= minimo and massimo <= 2 ** (n - 1) - 1):
        n += 1
    return n


def casi_peggiori(q: ReteIntera) -> dict[str, tuple[int, int]]:
    """Valore minimo e massimo che ogni grandezza può assumere con QUALSIASI ingresso.

    Per ogni neurone, il massimo si ottiene con gli ingressi al massimo dove il
    peso è positivo e a zero dove è negativo; il minimo nel caso opposto.
    Le somme parziali dell'accumulatore stanno sempre dentro questo intervallo,
    perché sono il bias più una parte dei prodotti.
    """
    h_max = 2 ** q.bit_h - 1
    acc1_max = int((np.maximum(q.W1, 0).sum(axis=0) * PIXEL_MAX + q.b1).max())
    acc1_min = int((np.minimum(q.W1, 0).sum(axis=0) * PIXEL_MAX + q.b1).min())
    acc2_max = int((np.maximum(q.W2, 0).sum(axis=0) * h_max + q.b2).max())
    acc2_min = int((np.minimum(q.W2, 0).sum(axis=0) * h_max + q.b2).min())
    w_max = 2 ** (BIT_PESI - 1) - 1
    return {
        "prodotto strato 1": (-w_max * PIXEL_MAX, w_max * PIXEL_MAX),
        "bias strato 1": (int(q.b1.min()), int(q.b1.max())),
        "acc1": (acc1_min, acc1_max),
        "prodotto strato 2": (-w_max * h_max, w_max * h_max),
        "bias strato 2": (int(q.b2.min()), int(q.b2.max())),
        "acc2": (acc2_min, acc2_max),
    }


def tronca_a_bit(a: np.ndarray, bit: int) -> np.ndarray:
    """Simula un registro con segno da `bit` bit: i valori fuori intervallo "girano" (overflow).

    Nel complemento a 2 troncare solo il risultato finale equivale a troncare
    ogni somma parziale, quindi basta applicarlo all'accumulatore completo.
    """
    meta = 2 ** (bit - 1)
    return (a + meta) % (2 * meta) - meta


# Autocontrollo, eseguito solo lanciando questo file
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = carica_dati()

    print("=== Divisione dei dati ===")
    print(f"addestramento: {len(X_train)} immagini   test: {len(X_test)} immagini   "
          f"totale: {len(X_train) + len(X_test)}")
    print("forma X_train:", X_train.shape, "  forma X_test:", X_test.shape)

    print("\n=== Pixel ===")
    print("tipo:", X_train.dtype, "  minimo:", X_train.min(),
          "  massimo:", max(X_train.max(), X_test.max()), f"(atteso {PIXEL_MAX})")

    print("\n=== Immagini per cifra ===")
    n_train = np.bincount(y_train)
    n_test = np.bincount(y_test)
    print("cifra  addestramento  test   % nel test")
    for cifra in range(10):
        totale = n_train[cifra] + n_test[cifra]
        print(f"  {cifra}       {n_train[cifra]:4d}       {n_test[cifra]:4d}    "
              f"{100 * n_test[cifra] / totale:5.1f} %")

    # Si interrompe qui se dimensioni o valori dei pixel non sono quelli attesi.
    assert len(X_train) + len(X_test) == 1797, "manca qualche immagine"
    assert X_train.max() <= PIXEL_MAX and X_test.max() <= PIXEL_MAX, "saturazione non applicata"
    assert X_train.min() >= 0 and X_test.min() >= 0, "pixel negativi"
    print("\nAutocontrollo superato.")
