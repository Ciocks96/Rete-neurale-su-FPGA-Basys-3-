"""Verifica del calcolo della rete e passaggio ai numeri interi (passi 6-9).

Nel passo 6 ricostruiamo il calcolo con numpy e lo confrontiamo con
predict() di scikit-learn. Nel passo 7 quantizziamo pesi e uscite:
predici_intero() in modello.py è il riferimento per il VHDL. Nel passo 8
proviamo meno bit per pesi e uscite nascoste, su 5 reti diverse. Nel passo 9
calcoliamo quanti bit servono agli accumulatori nel caso peggiore.
"""
import matplotlib.pyplot as plt
import numpy as np

from modello import (BIT_H, BIT_PESI, PIXEL_MAX, SCALA_INGRESSO, addestra_rete,
                     bit_con_segno, carica_dati, casi_peggiori, predici_intero,
                     quantizza_rete, salva_e_mostra_grafici, tronca_a_bit)

np.set_printoptions(precision=2, suppress=True, linewidth=120)

X_train, X_test, y_train, y_test = carica_dati()
rete = addestra_rete(X_train, y_train)          # rete 64-16-10, seme 0

# 1. Pesi della rete
# scikit-learn li tiene in coefs_ e intercepts_, uno per ogni strato.
# W1 e b1 sono del primo strato; W2 e b2 del secondo.
W1, W2 = rete.coefs_
b1, b2 = rete.intercepts_

print("=== 1. I pesi della rete 64-16-10 ===")
for nome, m in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
    print(f"{nome}: forma {str(m.shape):9s}  minimo {m.min():7.3f}  massimo {m.max():7.3f}  "
          f"valore assoluto massimo {np.abs(m).max():6.3f}")
print("(questi intervalli serviranno per scegliere la scala della quantizzazione)")


# 2. Calcolo della rete con numpy
def calcola_rete(X, W1, b1, W2, b2):
    """
    Calcola la rete per le immagini di X (una per riga).
    Restituisce la cifra prevista, le uscite nascoste h e i punteggi z2.

    Con @, ogni riga di X viene moltiplicata per i pesi dello strato.
    """
    z1 = X @ W1 + b1                 # primo strato
    h = np.maximum(z1, 0)            # ReLU
    z2 = h @ W2 + b2                 # secondo strato
    cifra = z2.argmax(axis=1)        # indice del punteggio più alto
    return cifra, h, z2


cifre_nostre, h, z2 = calcola_rete(X_test / SCALA_INGRESSO, W1, b1, W2, b2)
cifre_sklearn = rete.predict(X_test / SCALA_INGRESSO)
uguali = np.count_nonzero(cifre_nostre == cifre_sklearn)

print("\n=== 2. Il nostro calcolo contro predict() ===")
print(f"stessa cifra su {uguali} immagini su {len(X_test)}")
print(f"accuratezza del nostro calcolo: {np.mean(cifre_nostre == y_test) * 100:.2f} %")
assert uguali == len(X_test), "il calcolo a mano non coincide con predict()"


# 3. Un'immagine di test, vista strato per strato
k = 0                                            # prima immagine
x = X_test[k]
print(f"\n=== 3. L'immagine di test n. {k} (cifra vera: {y_test[k]}) ===")
print("pixel (8x8, valori 0-15):")
print(x.reshape(8, 8))
print("\nstrato 1, prima della ReLU (z1, 16 valori):")
print(x / SCALA_INGRESSO @ W1 + b1)
print("strato 1, dopo la ReLU (h, 16 valori; i negativi sono diventati 0):")
print(h[k])
print(f"neuroni spenti dalla ReLU: {np.count_nonzero(h[k] == 0)} su {len(h[k])}")
print("\nstrato 2, punteggi delle 10 cifre (z2):")
for cifra, punteggio in enumerate(z2[k]):
    segno = "  <-- il più alto" if cifra == cifre_nostre[k] else ""
    print(f"  cifra {cifra}: {punteggio:7.2f}{segno}")
print(f"cifra prevista: {cifre_nostre[k]}")


# 4. Spostare la divisione per 16 nei pesi
# (x / 16) @ W1 equivale a x @ (W1 / 16). Così possiamo usare i pixel interi
# direttamente, senza fare la divisione sulla scheda. Dividere per 16 = 2^4
# significa solo spostare la virgola di 4 posizioni.
W1_per_pixel_interi = W1 / SCALA_INGRESSO
cifre_assorbite, _, z2_assorbiti = calcola_rete(X_test, W1_per_pixel_interi, b1, W2, b2)

print("\n=== 4. Divisione per 16 assorbita nei pesi del primo strato ===")
print(f"stessa cifra su {np.count_nonzero(cifre_assorbite == cifre_nostre)} immagini su {len(X_test)}")
print(f"differenza massima tra i punteggi: {np.abs(z2_assorbiti - z2).max():.1e} "
      f"(solo arrotondamenti del computer)")


# Passo 7: quantizzazione
# Ogni peso viene scalato e arrotondato a un intero a 8 bit. Tra i due strati
# si applicano ReLU, shift e saturazione:
#   acc1 = pixel · W1 + b1                      (interi, ~17 bit)
#   h    = min(255, max(0, acc1) >> SHIFT)      ReLU + shift + saturazione -> 8 bit
#   acc2 = h · W2 + b2                          (interi)
#   cifra = posizione del valore più alto di acc2
# Lo shift divide per 2^SHIFT e riporta h nell'intervallo a 8 bit. Proviamo i
# valori possibili sul set di addestramento, senza usare il test per scegliere.
q, prove = quantizza_rete(rete, X_train, y_train)

print("\n" + "=" * 75)
print(f"=== 7a. Pesi interi a {BIT_PESI} bit ===")
# Scale a potenza di 2 (virgola fissa): scala = 2^n, con n bit frazionari.
n1, n2 = int(np.log2(q.scala1)), int(np.log2(q.scala2))
print(f"scala strato 1: 2^{n1} = {q.scala1:.0f}  (peso intero = round(peso reale / 16 × 2^{n1}), "
      f"cioè peso reale in formato Q{8 - (n1 - 4)}.{n1 - 4})")
print(f"scala strato 2: 2^{n2} = {q.scala2:.0f}  (peso intero = round(peso reale × 2^{n2}), "
      f"formato Q{8 - n2}.{n2})")
for nome, m in [("W1", q.W1), ("b1", q.b1), ("W2", q.W2), ("b2", q.b2)]:
    print(f"{nome}: forma {str(m.shape):9s}  da {m.min():6d} a {m.max():6d}")
print("esempio, i primi 8 pesi del neurone nascosto 0:")
print("  reali (÷16):", (rete.coefs_[0][:8, 0] / SCALA_INGRESSO).round(4))
print("  interi     :", q.W1[:8, 0])

print(f"\n=== 7b. Scelta dello SHIFT (sulle {len(X_train)} immagini di addestramento) ===")
print(" shift | accuratezza | uscite nascoste sature")
for shift, acc, sature in prove:
    if 2 <= shift <= 12:                                   # intervallo mostrato
        segno = "  <-- scelto" if shift == q.shift else ""
        print(f"  {shift:4d} | {acc:9.2f} % | {sature:8.2f} %{segno}")
print(f"(shift piccolo: valori grandi, molte uscite sature a {2 ** BIT_H - 1}; "
      f"shift grande: valori piccoli, si perde precisione)")

# 7c. Confronto tra rete float e rete intera sul test
cifre_int, h_int, acc1_int, acc2_int = predici_intero(q, X_test)
acc_float = np.mean(cifre_nostre == y_test) * 100
acc_int = np.mean(cifre_int == y_test) * 100
concordi = np.count_nonzero(cifre_int == cifre_nostre)

print(f"\n=== 7c. Rete in virgola mobile contro rete a interi (test, {len(y_test)} immagini) ===")
print(f"accuratezza float : {acc_float:6.2f} %  ({np.count_nonzero(cifre_nostre == y_test)} giuste)")
print(f"accuratezza interi: {acc_int:6.2f} %  ({np.count_nonzero(cifre_int == y_test)} giuste)")
print(f"stessa risposta float/interi su {concordi} immagini su {len(y_test)}")
print(f"uscite nascoste sature (= {2 ** BIT_H - 1}) sul test: {np.mean(h_int == 2 ** BIT_H - 1) * 100:.2f} %"
      f"   valore massimo di h: {h_int.max()}")
print(f"acc1 osservati: da {acc1_int.min()} a {acc1_int.max()}   "
      f"acc2 osservati: da {acc2_int.min()} a {acc2_int.max()}")

# Confronto con la scala libera (la scelta precedente)
q_libera, _ = quantizza_rete(rete, X_train, y_train, potenza_di_2=False)
cifre_libera = predici_intero(q_libera, X_test)[0]
print(f"con scala libera (scala1 = {q_libera.scala1:.1f}, scala2 = {q_libera.scala2:.1f}, "
      f"shift {q_libera.shift}): {np.mean(cifre_libera == y_test) * 100:.2f} %")

# 7d. La stessa immagine del passo 6, calcolata con interi
print(f"\n=== 7d. L'immagine di test n. {k}, con soli interi (cifra vera: {y_test[k]}) ===")
print("acc1 (strato 1, prima della ReLU):", acc1_int[k])
print(f"h    (dopo ReLU, >> {q.shift}, saturazione):", h_int[k])
print("     h reale del passo 6, per confronto:", h[k])
print(f"     h intero riportato alla scala reale (h × 2^{q.shift} / scala1):",
      (h_int[k] * 2 ** q.shift / q.scala1).round(2))
print("acc2 (punteggi delle 10 cifre):")
for cifra, punteggio in enumerate(acc2_int[k]):
    segno = "  <-- il più alto" if cifra == cifre_int[k] else ""
    print(f"  cifra {cifra}: {punteggio:7d}{segno}")
print(f"cifra prevista dalla rete a interi: {cifre_int[k]}")


# Passo 8: quanti bit servono davvero?
# Ripetiamo la quantizzazione con meno bit per i pesi e per le uscite nascoste h.
# Per ogni combinazione lo shift viene scelto di nuovo sui dati di addestramento.
# Usiamo 5 reti (semi 0-4) per non trarre conclusioni da una sola rete fortunata.
BIT_DA_PROVARE = [8, 6, 5, 4, 3, 2]      # con 2 bit i pesi valgono solo -1, 0, 1
SEMI = [0, 1, 2, 3, 4]
N_PESI = q.W1.size + q.W2.size           # 64x16 + 16x10 = 1184

reti = [rete] + [addestra_rete(X_train, y_train, seme=s) for s in SEMI[1:]]
acc_float_semi = np.array([np.mean(r.predict(X_test / SCALA_INGRESSO) == y_test) * 100 for r in reti])

# griglia[i, j, s] = accuratezza sul test con BIT_DA_PROVARE[i] bit per i pesi,
# BIT_DA_PROVARE[j] bit per h e la rete del seme SEMI[s]
griglia = np.zeros((len(BIT_DA_PROVARE), len(BIT_DA_PROVARE), len(SEMI)))
for i, bit_pesi in enumerate(BIT_DA_PROVARE):
    for j, bit_h in enumerate(BIT_DA_PROVARE):
        for s, r in enumerate(reti):
            qq, _ = quantizza_rete(r, X_train, y_train, bit_pesi, bit_h)
            cifre, _, _, _ = predici_intero(qq, X_test)
            griglia[i, j, s] = np.mean(cifre == y_test) * 100
media = griglia.mean(axis=2)

print("\n" + "=" * 75)
print(f"=== 8. Quanti bit servono? (test, media su {len(SEMI)} reti) ===")
print(f"rete in virgola mobile: media {acc_float_semi.mean():.2f} %  "
      f"(per seme: {'  '.join(f'{a:.2f}' for a in acc_float_semi)})")

# 8a. Un parametro alla volta, l'altro resta a 8 bit
i8 = BIT_DA_PROVARE.index(8)
print("\n8a. Bit dei pesi (h a 8 bit)            |  Bit di h (pesi a 8 bit)")
print(" bit | media    min    max  | ROM pesi  |  media    min    max")
for k, bit in enumerate(BIT_DA_PROVARE):
    p, h_ = griglia[k, i8], griglia[i8, k]
    print(f"  {bit}  | {p.mean():6.2f} {p.min():6.2f} {p.max():6.2f} | {N_PESI * bit / 8:5.0f} B  | "
          f"{h_.mean():6.2f} {h_.min():6.2f} {h_.max():6.2f}")

# 8b. Tutte le combinazioni
print("\n8b. Accuratezza media (righe: bit dei pesi, colonne: bit di h)")
print("pesi\\h " + "".join(f"{b:>8d}" for b in BIT_DA_PROVARE))
for i, bit_pesi in enumerate(BIT_DA_PROVARE):
    print(f"  {bit_pesi:>3d}  " + "".join(f"{v:8.2f}" for v in media[i]))
print(f"(una differenza sotto ~1 punto è rumore: 1 immagine su {len(y_test)} vale "
      f"{100 / len(y_test):.2f} punti)")

# Grafico: le due curve di 8a
plt.figure(figsize=(7, 4))
for etichetta, dati in [("bit dei pesi (h a 8 bit)", griglia[:, i8]),
                        ("bit di h (pesi a 8 bit)", griglia[i8, :])]:
    m = dati.mean(axis=1)
    plt.errorbar(BIT_DA_PROVARE, m, yerr=[m - dati.min(axis=1), dati.max(axis=1) - m],
                 marker="o", capsize=4, label=etichetta)
plt.axhline(acc_float_semi.mean(), color="gray", linestyle="--", label="virgola mobile (media)")
plt.gca().invert_xaxis()
plt.xlabel("bit")
plt.ylabel("accuratezza sul test (%)")
plt.title(f"Accuratezza al variare dei bit ({len(SEMI)} reti; barre: min e max)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
salva_e_mostra_grafici("quantizza")


# Passo 9: quanti bit servono agli accumulatori?
# Un accumulatore troppo stretto non dà errori in hardware: il valore "gira"
# (overflow) e il risultato è sbagliato senza che nessuno se ne accorga.
# Per questo lo dimensioniamo sul caso peggiore possibile, non solo su quelli
# osservati. Nella versione parallela i MAC 0..9 vengono riusati per lo
# strato 2, quindi lo stesso accumulatore deve bastare per acc1 e per acc2.
print("\n" + "=" * 75)
print("=== 9a. Bit necessari (con segno, complemento a 2) ===")
peggiori = casi_peggiori(q)
X_tutte = np.concatenate([X_train, X_test])
_, h_tutte, acc1_tutte, acc2_tutte = predici_intero(q, X_tutte)
osservati = {"acc1": (int(acc1_tutte.min()), int(acc1_tutte.max())),
             "acc2": (int(acc2_tutte.min()), int(acc2_tutte.max()))}
print(f"{'grandezza':18s} | {'caso peggiore':>22s} | bit | {'osservato (1797 immagini)':>26s} | bit")
for nome, (lo, hi) in peggiori.items():
    riga = f"{nome:18s} | {lo:>10d} .. {hi:>8d} | {bit_con_segno(lo, hi):3d} |"
    if nome in osservati:
        olo, ohi = osservati[nome]
        riga += f" {olo:>12d} .. {ohi:>9d} | {bit_con_segno(olo, ohi):3d}"
    print(riga)
BIT_ACC = max(bit_con_segno(*peggiori["acc1"]), bit_con_segno(*peggiori["acc2"]))
print(f"accumulatore condiviso tra i due strati: {BIT_ACC} bit")
# Limite valido per qualsiasi peso a 8 bit (se si riaddestra la rete):
lim1 = 64 * 127 * PIXEL_MAX + int(np.abs(q.b1).max())
lim2 = 16 * 127 * (2 ** BIT_H - 1) + int(np.abs(q.b2).max())
print(f"(limite per pesi qualsiasi a {BIT_PESI} bit: acc1 {bit_con_segno(-lim1, lim1)} bit, "
      f"acc2 {bit_con_segno(-lim2, lim2)} bit)")

# 9b. Prova pratica: cosa succede se l'accumulatore ha meno bit?
print("\n=== 9b. Accumulatore troppo stretto: accuratezza sul test ===")
print(" bit acc | accuratezza | immagini con overflow")
for bit in range(BIT_ACC - 6, BIT_ACC + 1):
    acc1_t = tronca_a_bit(acc1_int, bit)
    h_t = np.minimum(2 ** q.bit_h - 1, np.maximum(acc1_t, 0) >> q.shift)
    acc2_t = tronca_a_bit(h_t @ q.W2 + q.b2, bit)
    errate = np.any(acc1_t != acc1_int, axis=1) | np.any(acc2_t != h_t @ q.W2 + q.b2, axis=1)
    acc_t = np.mean(acc2_t.argmax(axis=1) == y_test) * 100
    segno = "  <-- scelto" if bit == BIT_ACC else ""
    print(f"   {bit:3d}  |  {acc_t:7.2f} %  | {np.count_nonzero(errate):4d}{segno}")
