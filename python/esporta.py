"""Esporta la rete a interi per il VHDL (passo 10).

Genera nella cartella vhdl/ del progetto:
  - rete_pkg.vhd     costanti, pesi e bias della rete. I pesi sono organizzati
                     per riga, perché nella versione parallela i MAC leggono
                     tutti i pesi di una riga nello stesso ciclo.
  - vettori_test.txt immagini di test con i risultati attesi, per il testbench.

Alla fine rilegge i file generati e controlla che contengano gli stessi
numeri della rete in Python.
"""
from pathlib import Path

import numpy as np

from modello import (BIT_H, BIT_PESI, PIXEL_MAX, ReteIntera, addestra_rete,
                     bit_con_segno, carica_dati, casi_peggiori, predici_intero,
                     quantizza_rete)

CARTELLA_VHDL = Path(__file__).resolve().parent.parent / "vhdl"
BIT_PIXEL = 4                       # pixel 0..15

X_train, X_test, y_train, y_test = carica_dati()
rete = addestra_rete(X_train, y_train)              # rete 64-16-10, seme 0
q, _ = quantizza_rete(rete, X_train, y_train)

N_IN, N_HID = q.W1.shape                            # 64, 16
N_OUT = q.W2.shape[1]                               # 10
peggiori = casi_peggiori(q)
BIT_ACC = max(bit_con_segno(*peggiori["acc1"]), bit_con_segno(*peggiori["acc2"]))
assert PIXEL_MAX < 2 ** BIT_PIXEL


# 1. Impacchettare una riga di pesi in un vettore di bit
# Il peso della colonna j occupa i bit 8j+7 .. 8j (colonna 0 nei bit più bassi).
# I numeri negativi sono scritti in complemento a 2: -5 a 8 bit è 0xFB.
def riga_esadecimale(pesi: np.ndarray, bit: int) -> str:
    """Restituisce la riga come stringa esadecimale; la colonna 0 è a destra."""
    valore = 0
    for j, w in enumerate(pesi):
        valore |= (int(w) & (2 ** bit - 1)) << (bit * j)
    cifre_hex = len(pesi) * bit // 4
    return f"{valore:0{cifre_hex}X}"


def leggi_riga(testo: str, n: int, bit: int) -> list[int]:
    """Operazione inversa di riga_esadecimale, usata per il controllo finale."""
    valore = int(testo, 16)
    pesi = []
    for j in range(n):
        w = (valore >> (bit * j)) & (2 ** bit - 1)
        pesi.append(w - 2 ** bit if w >= 2 ** (bit - 1) else w)   # complemento a 2
    return pesi


# 2. Package VHDL
def tabella_righe(nome: str, W: np.ndarray, commento: str) -> list[str]:
    n_righe, n_col = W.shape
    righe = [f"  -- {commento}",
             f"  type {nome}_t is array (0 to {n_righe - 1}) of "
             f"std_logic_vector({n_col} * PESO_BIT - 1 downto 0);",
             f"  constant {nome} : {nome}_t := ("]
    for k in range(n_righe):
        fine = "," if k < n_righe - 1 else ""
        valori = " ".join(f"{int(w):4d}" for w in W[k])
        righe.append(f'    x"{riga_esadecimale(W[k], BIT_PESI)}"{fine}  -- {k:2d}: {valori}')
    righe.append("  );")
    return righe


def tabella_bias(nome: str, b: np.ndarray, n: str) -> list[str]:
    valori = ", ".join(str(int(v)) for v in b)
    return [f"  type {nome}_t is array (0 to {n} - 1) of integer;",
            f"  constant {nome} : {nome}_t := ({valori});"]


def scrivi_package(q: ReteIntera, file: Path) -> None:
    testo = [
        "-- File generato da python/esporta.py: non modificarlo a mano.",
        "-- Per cambiare la rete si modifica il Python e si rilancia esporta.py.",
        "--",
        "-- Calcolo (identico a predici_intero() in python/modello.py):",
        "--   acc1[j] = B1[j] + somma_k pixel[k] * W1[k][j]       k = 0..63, j = 0..15",
        "--   h[j]    = min(255, max(0, acc1[j]) >> SHIFT)",
        "--   acc2[c] = B2[c] + somma_j h[j] * W2[j][c]           c = 0..9",
        "--   cifra   = indice del massimo di acc2 (a parità vince l'indice più basso)",
        "-- Pixel in ordine riga per riga: il pixel (r, c) dell'immagine 8x8 è k = 8r + c.",
        "-- Pesi in complemento a 2; nella riga k, la colonna j occupa i bit",
        "-- 8j+7 .. 8j (colonna 0 a destra nel valore esadecimale).",
        "",
        "library ieee;",
        "use ieee.std_logic_1164.all;",
        "",
        "package rete_pkg is",
        "",
        "  -- Dimensioni",
        f"  constant N_IN      : integer := {N_IN};    -- pixel",
        f"  constant N_HID     : integer := {N_HID};    -- neuroni nascosti (= MAC in parallelo)",
        f"  constant N_OUT     : integer := {N_OUT};    -- cifre",
        "",
        "  -- Bit",
        f"  constant PIXEL_BIT : integer := {BIT_PIXEL};     -- senza segno, 0..{PIXEL_MAX}",
        f"  constant PESO_BIT  : integer := {BIT_PESI};     -- con segno",
        f"  constant H_BIT     : integer := {q.bit_h};     -- senza segno, 0..{2 ** q.bit_h - 1}",
        f"  constant ACC_BIT   : integer := {BIT_ACC};    -- con segno, caso peggiore (passo 9)",
        f"  constant SHIFT     : integer := {q.shift};     -- h = acc1 >> SHIFT",
        "",
        *tabella_righe("W1", q.W1, f"Strato 1: riga k = i {N_HID} pesi del pixel k"),
        "",
        *tabella_righe("W2", q.W2, f"Strato 2: riga j = i {N_OUT} pesi dell'uscita nascosta h[j]"),
        "",
        "  -- Bias, già alla scala degli accumulatori",
        *tabella_bias("B1", q.b1, "N_HID"),
        *tabella_bias("B2", q.b2, "N_OUT"),
        "",
        "end package rete_pkg;",
        "",
    ]
    file.write_text("\n".join(testo))


# 3. Vettori di test
# Una riga per immagine, numeri decimali separati da spazi (facili da leggere
# in VHDL con textio): 64 pixel, 16 h attesi, 10 punteggi attesi, cifra
# prevista dalla rete a interi, cifra vera.
def scrivi_vettori(q: ReteIntera, X: np.ndarray, y: np.ndarray, file: Path) -> None:
    cifre, h, _, acc2 = predici_intero(q, X)
    with file.open("w") as f:
        for i in range(len(X)):
            numeri = [*X[i], *h[i], *acc2[i], cifre[i], y[i]]
            f.write(" ".join(str(int(v)) for v in numeri) + "\n")


CARTELLA_VHDL.mkdir(exist_ok=True)
file_pkg = CARTELLA_VHDL / "rete_pkg.vhd"
file_vettori = CARTELLA_VHDL / "vettori_test.txt"
scrivi_package(q, file_pkg)
scrivi_vettori(q, X_test, y_test, file_vettori)

print("=== 10. Esportazione ===")
print(f"rete 64-{N_HID}-{N_OUT}: pesi a {BIT_PESI} bit, h a {BIT_H} bit, "
      f"SHIFT = {q.shift}, accumulatore a {BIT_ACC} bit")
print(f"ROM strato 1: {N_IN} righe da {N_HID * BIT_PESI} bit   "
      f"ROM strato 2: {N_HID} righe da {N_OUT * BIT_PESI} bit")
print(f"scritto: vhdl/{file_pkg.name}  ({len(file_pkg.read_text().splitlines())} righe)")
print(f"scritto: vhdl/{file_vettori.name}  ({len(X_test)} immagini)")


# 4. Controllo: rileggiamo i file e ricalcoliamo la rete da lì
testo = file_pkg.read_text()
righe_hex = [r.split('"')[1] for r in testo.splitlines() if r.strip().startswith('x"')]
W1_letto = np.array([leggi_riga(r, N_HID, BIT_PESI) for r in righe_hex[:N_IN]])
W2_letto = np.array([leggi_riga(r, N_OUT, BIT_PESI) for r in righe_hex[N_IN:]])
assert np.array_equal(W1_letto, q.W1) and np.array_equal(W2_letto, q.W2), "pesi diversi"

dati = np.loadtxt(file_vettori, dtype=np.int64)
pixel, h_att = dati[:, :N_IN], dati[:, N_IN:N_IN + N_HID]
acc2_att = dati[:, N_IN + N_HID:N_IN + N_HID + N_OUT]
cifra_att, cifra_vera = dati[:, -2], dati[:, -1]
q_letta = ReteIntera(W1_letto, q.b1, W2_letto, q.b2, q.shift, q.bit_h, q.scala1, q.scala2)
cifre, h, _, acc2 = predici_intero(q_letta, pixel)
assert np.array_equal(h, h_att) and np.array_equal(acc2, acc2_att), "vettori diversi"
assert np.array_equal(cifre, cifra_att), "cifre diverse"

print("\ncontrollo: pesi riletti dal package = pesi Python; "
      "rete ricalcolata dai file = valori attesi")
print(f"accuratezza attesa in hardware: {np.mean(cifra_att == cifra_vera) * 100:.2f} % "
      f"({np.count_nonzero(cifra_att == cifra_vera)} su {len(cifra_vera)})")
print(f"\nesempio, riga 0 di W1 (pesi del pixel 0 per i 16 neuroni): x\"{righe_hex[0]}\"")
print(f"esempio, riga 1 di W1: x\"{righe_hex[1]}\" -> {leggi_riga(righe_hex[1], N_HID, BIT_PESI)}")
