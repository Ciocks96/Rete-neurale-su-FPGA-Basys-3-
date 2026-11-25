"""Addestra e valuta la rete 64 -> N_NASCOSTI -> 10, e confronta alcune configurazioni."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# La configurazione e la funzione di addestramento sono condivise con quantizza.py.
from modello import N_NASCOSTI, SCALA_INGRESSO, addestra_rete, carica_dati, salva_e_mostra_grafici

SEMI = [0, 1, 2, 3, 4]          # semi usati nei confronti
NEURONI_DA_PROVARE = [4, 8, 16, 32, 64]   # dimensioni da provare (passo 5)
F_CLOCK_MHZ = 100               # clock della Basys 3


def cicli_di_clock(n_nascosti):
    """
    Stima dei cicli con un solo moltiplicatore (un MAC per ciclo).
    Per ogni neurone si conta un ciclo per il bias, uno per ingresso e uno
    per salvare il risultato. Il conteggio esatto dipenderà dal VHDL.
    """
    return n_nascosti * (1 + 64 + 1) + 10 * (1 + n_nascosti + 1)


def errori_frequenti(vere, previste, quanti=5):
    """Stampa le coppie (cifra vera -> cifra prevista) sbagliate più spesso."""
    sbagliate = vere != previste
    coppie, conteggi = np.unique(np.stack([vere[sbagliate], previste[sbagliate]], axis=1),
                                 axis=0, return_counts=True)
    for (vera, prevista), n in sorted(zip(map(tuple, coppie), conteggi), key=lambda t: -t[1])[:quanti]:
        print(f"  {vera} -> {prevista}: {n} {'volta' if n == 1 else 'volte'}")


X_train, X_test, y_train, y_test = carica_dati()

# Rete scelta, addestrata con il seme 0
rete = addestra_rete(X_train, y_train, N_NASCOSTI, seme=0)
acc_train = rete.score(X_train / SCALA_INGRESSO, y_train)
acc_test = rete.score(X_test / SCALA_INGRESSO, y_test)
previste = rete.predict(X_test / SCALA_INGRESSO)

# coefs_ contiene i pesi dei due strati.
pesi = sum(w.size for w in rete.coefs_)
bias = sum(b.size for b in rete.intercepts_)

print(f"=== Rete 64 -> {N_NASCOSTI} (ReLU) -> 10, seme 0 ===")
print(f"accuratezza addestramento: {acc_train * 100:6.2f} %")
print(f"accuratezza test         : {acc_test * 100:6.2f} %   "
      f"({np.count_nonzero(previste != y_test)} errori su {len(y_test)})")
print(f"pesi: 64x{N_NASCOSTI} + {N_NASCOSTI}x10 = {pesi} (+ {bias} bias)   "
      f"moltiplicazioni per immagine: {pesi}")
print("errori più frequenti (vera -> prevista):")
errori_frequenti(y_test, previste)

# Ripetiamo l'addestramento con semi diversi e confrontiamo i pixel saturati
# (0-15) con quelli originali (0-16).
print(f"\n=== Variabilità con {len(SEMI)} semi, e verifica della saturazione ===")
X16_train, X16_test, y16_train, y16_test = carica_dati(pixel_max=None)   # senza saturazione

risultati = {}
for nome, (Xtr, Xte, ytr, yte) in {
        "pixel 0-15 (saturati)": (X_train, X_test, y_train, y_test),
        "pixel 0-16 (originali)": (X16_train, X16_test, y16_train, y16_test)}.items():
    accuratezze = []
    for seme in SEMI:
        r = addestra_rete(Xtr, ytr, N_NASCOSTI, seme)
        accuratezze.append(r.score(Xte / SCALA_INGRESSO, yte) * 100)
    a = np.array(accuratezze)
    risultati[nome] = a
    print(f"{nome:24s}: " + "  ".join(f"{v:5.2f}" for v in a) +
          f"   media {a.mean():5.2f} %  (min {a.min():5.2f}, max {a.max():5.2f}, "
          f"deviazione std {a.std():4.2f})")

diff = risultati["pixel 0-15 (saturati)"].mean() - risultati["pixel 0-16 (originali)"].mean()
print(f"differenza media saturati - originali: {diff:+.2f} punti "
      f"(= {diff / 100 * len(y_test):+.1f} immagini su {len(y_test)})")

# Passo 5: accuratezza e costo al variare dei neuroni nascosti
print(f"\n=== Passo 5: quanti neuroni nascosti? ({len(SEMI)} semi per ognuno, pixel 0-15) ===")
print(f"{'neuroni':>7} | {'media':>7} {'min':>6} {'max':>6} {'dev.std':>7} | "
      f"{'pesi':>6} {'ROM 8 bit':>10} | {'cicli':>6} {'tempo':>8}")
confronto = []
for n in NEURONI_DA_PROVARE:
    acc = np.array([addestra_rete(X_train, y_train, n, seme).score(X_test / SCALA_INGRESSO, y_test) * 100
                    for seme in SEMI])
    n_pesi = 64 * n + n * 10
    cicli = cicli_di_clock(n)
    confronto.append((n, acc))
    print(f"{n:>7} | {acc.mean():6.2f}% {acc.min():6.2f} {acc.max():6.2f} {acc.std():7.2f} | "
          f"{n_pesi:>6} {n_pesi / 1024:7.2f} KB | {cicli:>6} {cicli / F_CLOCK_MHZ:6.1f} µs")
print("(ROM 8 bit = pesi a 1 byte ciascuno; tempo = cicli a", F_CLOCK_MHZ, "MHz con un solo moltiplicatore;")
print(" per confronto, ricevere un'immagine dalla seriale richiede circa 5600 µs)")

# Accuratezza media e intervallo min-max per ogni dimensione.
plt.figure(figsize=(7, 4))
neuroni = [n for n, _ in confronto]
medie = [a.mean() for _, a in confronto]
plt.errorbar(neuroni, medie,
             yerr=[[a.mean() - a.min() for _, a in confronto], [a.max() - a.mean() for _, a in confronto]],
             marker="o", capsize=4)
plt.xscale("log", base=2)
plt.xticks(neuroni, [str(n) for n in neuroni])
plt.xlabel("neuroni nascosti")
plt.ylabel("accuratezza sul test (%)")
plt.title(f"Accuratezza media su {len(SEMI)} semi (barre: minimo e massimo)")
plt.grid(alpha=0.3)
plt.tight_layout()

# Matrice di confusione: righe = cifre vere, colonne = cifre previste.
plt.figure(figsize=(5.5, 5))
ConfusionMatrixDisplay.from_predictions(y_test, previste, cmap="Blues", colorbar=False, ax=plt.gca())
plt.title(f"Rete 64-{N_NASCOSTI}-10 – test {acc_test * 100:.1f} %")
plt.xlabel("cifra prevista")
plt.ylabel("cifra vera")
plt.tight_layout()
salva_e_mostra_grafici("addestra")
