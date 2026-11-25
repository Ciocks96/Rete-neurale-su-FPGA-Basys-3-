# Rete neurale su FPGA (Basys 3)

Progetto per il corso di Sistemi Embedded: riconoscimento di cifre scritte a mano con una rete neurale eseguita su una scheda **Digilent Basys 3** (Xilinx Artix-7 XC7A35T, Vivado).

## La rete

- dataset `digits` di scikit-learn: 1797 immagini 8×8, pixel saturati a 15 (4 bit)
- rete 64 → 16 (ReLU) → 10, 1184 pesi
- quantizzazione a interi: pesi a 8 bit in virgola fissa (Q2.6 e Q3.5), uscite nascoste a 8 bit, accumulatori a 18 bit
- accuratezza sul test: **97,11%**, identica tra rete in virgola mobile e rete a interi
- hardware previsto: versione parallela con 16 MAC (un MAC per neurone nascosto)

## Struttura

```
python/      modello e script, in ordine di uso:
             modello.py      funzioni condivise (dati, addestramento, quantizzazione, golden model)
             esplora_dati.py esplorazione del dataset
             addestra.py     addestramento e scelta del numero di neuroni
             quantizza.py    verifica del calcolo, quantizzazione, bit necessari
             esporta.py      genera i file per il VHDL
vhdl/        rete_pkg.vhd (costanti e pesi) e vettori_test.txt (risultati attesi), generati da esporta.py
schemi/      schemi a blocchi del modello (PDF/PNG), rigenerabili con genera_schemi.py
diario_progetto.md   diario con passi, scelte e misure
```

## Come eseguire

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd python
python quantizza.py      # oppure addestra.py, esporta.py, ...
```
