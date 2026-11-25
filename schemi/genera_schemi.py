"""Genera schemi/schema_modello.html (due figure SVG) per il progetto."""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

# Colori
C = {
    "dati":   ("#e8f1fb", "#2f6db5"),   # riempimento, bordo
    "op":     ("#fdf0e2", "#c46a12"),
    "par":    ("#e7f5ea", "#2e8a46"),
    "file":   ("#f1ecfa", "#6a46b0"),
    "futuro": ("#f4f4f4", "#8a8a8a"),
    "nota":   ("#fffbe6", "#b59a00"),
}
TXT = "#1d1d1f"
SUB = "#55585e"


def box(x, y, w, h, kind, title, lines=(), title_size=17, line_size=13.5, dashed=False,
        mono_from=None):
    fill, stroke = C[kind]
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="2"{dash}/>']
    s.append(f'<text x="{x + 14}" y="{y + 26}" font-size="{title_size}" font-weight="700" '
             f'fill="{stroke}">{escape(title)}</text>')
    yy = y + 26 + line_size * 1.65
    for i, ln in enumerate(lines):
        mono = mono_from is not None and i >= mono_from
        fam = ' font-family="Menlo, Consolas, monospace"' if (mono or ln.startswith("§")) else ""
        ln = ln.lstrip("§")
        bold = ""
        if ln.startswith("**"):
            ln = ln.strip("*"); bold = ' font-weight="700"'
        s.append(f'<text x="{x + 14}" y="{yy}" font-size="{line_size}" fill="{TXT}"{fam}{bold}>'
                 f'{escape(ln)}</text>')
        yy += line_size * 1.5
    return "\n".join(s)


def arrow(x1, y1, x2, y2, label=None, color="#44474d", dashed=False, lx=None, ly=None,
          anchor="middle"):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    s = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2" '
         f'marker-end="url(#freccia)"{dash}/>']
    if label:
        lx = (x1 + x2) / 2 if lx is None else lx
        ly = (y1 + y2) / 2 - 8 if ly is None else ly
        s.append(f'<text x="{lx}" y="{ly}" font-size="12.5" fill="{SUB}" text-anchor="{anchor}" '
                 f'font-style="italic">{escape(label)}</text>')
    return "\n".join(s)


def path_arrow(d, color="#44474d", dashed=False):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" '
            f'marker-end="url(#freccia)"{dash}/>')


def text(x, y, t, size=13, color=TXT, weight="400", anchor="start", mono=False, italic=False):
    fam = ' font-family="Menlo, Consolas, monospace"' if mono else ""
    it = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{fam}{it}>{escape(t)}</text>')


DEFS = """<defs><marker id="freccia" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#44474d"/></marker></defs>"""


def legenda(x, y, voci):
    s = []
    for i, (kind, nome) in enumerate(voci):
        fill, stroke = C[kind]
        xx = x + i * 205
        dash = ' stroke-dasharray="5 4"' if kind == "futuro" else ""
        s.append(f'<rect x="{xx}" y="{y}" width="22" height="16" rx="4" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="2"{dash}/>')
        s.append(text(xx + 30, y + 13, nome, 13, SUB))
    return "\n".join(s)


# ---------------------------------------------------------------- Figura 1
W, H = 1400, 900
f1 = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
      f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">', DEFS]

# Riga 1: dati e addestramento
f1.append(box(30, 40, 300, 190, "dati", "Dataset digits", [
    "scikit-learn, load_digits()",
    "1797 immagini 8×8",
    "pixel interi 0..16",
    "10 cifre, ~180 immagini ciascuna",
    "§d.data: 1797 × 64 (riga per riga)",
]))
f1.append(arrow(330, 135, 395, 135))
f1.append(box(400, 40, 330, 190, "op", "carica_dati()   modello.py", [
    "1. pixel → interi",
    "2. saturazione: 16 → 15 (4 bit, 0..15)",
    "3. divisione stratificata 75 / 25",
    "   seme fisso SEME = 0",
    "§X_train 1347×64   X_test 450×64",
]))
f1.append(arrow(730, 135, 795, 135))
f1.append(box(800, 40, 330, 190, "op", "addestra_rete()   modello.py", [
    "MLPClassifier 64 → 16 (ReLU) → 10",
    "ingressi ÷ 16 (SCALA_INGRESSO)",
    "max_iter = 3000, seme rete = 0",
    "scelte: addestra.py (passi 4–5)",
    "**test: 97,11 %  (media 5 semi 97,29 %)",
]))
f1.append(arrow(1130, 135, 1185, 135))
f1.append(box(1190, 40, 180, 190, "par", "Pesi float", [
    "§W1  64 × 16",
    "§b1  16",
    "§W2  16 × 10",
    "§b2  10",
    "1210 numeri reali",
]))

# freccia di ritorno verso la riga 2
f1.append(path_arrow("M1280,230 L1280,262 L120,262 L120,292"))
f1.append(text(700, 255, "i pesi reali vengono convertiti in interi", 12.5, SUB, anchor="middle",
               italic=True))

# Riga 2: quantizzazione
f1.append(box(30, 295, 440, 265, "op", "quantizza_rete()   modello.py (passo 7)", [
    "pesi → interi a 8 bit, scala a potenza di 2:",
    "§W1q = round(W1 / 16 × 2^10)   Q2.6   −108..108",
    "§b1q = round(b1 × 2^10)               −252..600",
    "§W2q = round(W2 × 2^5)         Q3.5   −81..59",
    "scelta dello SHIFT sui dati di addestramento:",
    "  prova 0..20 → max accuratezza, meno sature",
    "§SHIFT = 6",
    "§b2q = round(b2 × 2^5 × 2^10 / 2^6)   −309..382",
]))
f1.append(arrow(470, 427, 525, 427))
f1.append(box(530, 295, 400, 265, "op", "predici_intero()   modello.py", [
    "**il GOLDEN MODEL: ciò che farà la FPGA",
    "§acc1 = x · W1q + b1q",
    "§h    = min(255, max(0, acc1) >> 6)",
    "§acc2 = h · W2q + b2q",
    "§cifra = argmax(acc2)",
    "solo interi, nessuna scala in hardware",
    "**interi 97,11 % = float 97,11 %",
    "stessa risposta su 449 / 450",
]))
f1.append(arrow(930, 427, 985, 427))
f1.append(box(990, 295, 380, 265, "op", "casi_peggiori()   modello.py (passo 9)", [
    "min / max possibili con qualsiasi immagine",
    "§prodotto 1   −1905..1905       12 bit",
    "§acc1       −16333..20985       16 bit",
    "§prodotto 2  −32385..32385      16 bit",
    "§acc2       −86318..48141       18 bit",
    "MAC riusati nei due strati:",
    "**ACC_BIT = max(16, 18) = 18 bit",
    "prova bit 8..2 (passo 8): restiamo a 8",
]))

# Riga 3: esportazione
f1.append(path_arrow("M750,560 L750,612"))
f1.append(path_arrow("M1180,560 L1180,585 L860,585 L860,612"))
f1.append(box(530, 615, 400, 120, "op", "esporta.py   (passo 10)", [
    "riaddestra (seme 0) e quantizza la stessa rete",
    "scrive i file e li RILEGGE per controllo:",
    "pesi riletti = pesi Python, rete ricalcolata = attesa",
]))
f1.append(path_arrow("M600,735 L600,760 L250,760 L250,782"))
f1.append(path_arrow("M860,735 L860,760 L1150,760 L1150,782"))
f1.append(box(30, 785, 470, 100, "file", "vhdl/rete_pkg.vhd", [
    "costanti: N_IN 64, N_HID 16, N_OUT 10, PIXEL_BIT 4,",
    "PESO_BIT 8, H_BIT 8, ACC_BIT 18, SHIFT 6",
    "W1: 64 righe × 128 bit · W2: 16 righe × 80 bit · B1, B2",
], line_size=13))
f1.append(box(900, 785, 470, 100, "file", "vhdl/vettori_test.txt", [
    "450 righe, una per immagine di test:",
    "§64 pixel | 16 h | 10 acc2 | cifra prevista | vera",
    "il testbench deve ritrovarli bit per bit",
], line_size=13))
f1.append(box(560, 795, 300, 80, "futuro", "VHDL (prossimo)", [
    "rete parallela, 16 MAC",
    "testbench: confronto con i vettori",
], dashed=True))
f1.append(arrow(500, 835, 555, 835, dashed=True))
f1.append(arrow(900, 835, 865, 835, dashed=True))

f1.append(legenda(30, 10, [("dati", "dati"), ("op", "elaborazione in Python"),
                            ("par", "parametri"), ("file", "file generati"),
                            ("futuro", "da fare")]))
f1.append("</svg>")

# ---------------------------------------------------------------- Figura 2
pix = [int(v) for v in "0 0 7 15 14 3 0 0 0 0 9 14 11 15 0 0 0 0 1 5 0 15 5 0 0 0 0 0 0 15 5 0 0 0 0 0 3 15 4 0 0 0 0 1 12 14 1 0 0 0 5 12 15 15 14 1 0 0 8 15 14 10 13 3".split()]
acc1 = [6928, -787, 739, -1902, 1515, -335, -567, 221, 2920, -1490, -38, -908, 4667, 1464, 3152, 2603]
h = [108, 0, 11, 0, 23, 0, 0, 3, 45, 0, 0, 0, 72, 22, 49, 40]
acc2 = [711, -1696, 3377, 1653, -7087, -2708, -622, -1192, -2688, -1234]

W, H = 1400, 940
f2 = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
      f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">', DEFS]

# Immagine 8x8
gx, gy, cs = 40, 95, 24
f2.append(text(gx, 62, "Ingresso: immagine di test n. 0", 16, C["dati"][1], "700"))
f2.append(text(gx, 82, "(cifra vera: 2)", 13, SUB))
for r in range(8):
    for c in range(8):
        v = pix[8 * r + c]
        g = int(255 - v * 255 / 15)
        col = "#ffffff" if v < 8 else "#000000"
        col = "#000" if v < 8 else "#fff"
        f2.append(f'<rect x="{gx + c * cs}" y="{gy + r * cs}" width="{cs}" height="{cs}" '
                  f'fill="rgb({g},{g},{g})" stroke="#c9c9c9" stroke-width="0.6"/>')
        f2.append(f'<text x="{gx + c * cs + cs / 2}" y="{gy + r * cs + 16}" font-size="10" '
                  f'text-anchor="middle" fill="{col}">{v}</text>')
f2.append(text(gx, gy + 8 * cs + 22, "64 pixel, 4 bit senza segno (0..15)", 13))
f2.append(text(gx, gy + 8 * cs + 40, "ordine riga per riga: k = 8·r + c", 13, SUB))
f2.append(text(gx, gy + 8 * cs + 58, "§x[0..63]".lstrip("§"), 13, TXT, mono=True))

# Strato 1
f2.append(arrow(240, 190, 290, 190, "x[k]", ly=180))
f2.append(box(295, 40, 330, 330, "op", "Strato 1: 16 neuroni", [
    "per ogni neurone j = 0..15:",
    "§acc1[j] = B1[j] + Σ x[k]·W1[k][j]",
    "§          k = 0..63",
    "",
    "64 prodotti pixel × peso:",
    "  4 bit × 8 bit con segno → 12 bit",
    "somma in un accumulatore:",
    "  18 bit con segno (caso peggiore 16)",
    "",
    "in hardware: 16 MAC in parallelo,",
    "1 pixel per ciclo → 64 cicli",
]))
f2.append(box(295, 390, 330, 120, "par", "ROM W1 + bias B1", [
    "64 righe × 16 pesi da 8 bit (128 bit)",
    "riga k = pesi del pixel k per i 16 neuroni",
    "§B1 = 600, 125, 372, … (16 valori)",
]))
f2.append(arrow(460, 390, 460, 375))

# Riquantizzazione
f2.append(arrow(625, 190, 675, 190, "acc1[j]", ly=180))
f2.append(box(680, 40, 290, 330, "op", "Tra gli strati", [
    "1. ReLU: negativo → 0",
    "2. shift: >> 6  (÷ 64, solo fili)",
    "3. saturazione: > 255 → 255",
    "",
    "§h[j] = min(255,",
    "§         max(0, acc1[j]) >> 6)",
    "",
    "18 bit con segno → 8 bit senza segno",
    "",
    "**neurone 0 dell'immagine 0:",
    "§6928 → ReLU → 6928",
    "§     → >>6  → 108",
    "§     → sat  → 108  (< 255)",
]))

# Strato 2
f2.append(arrow(970, 190, 1020, 190, "h[j]", ly=180))
f2.append(box(1025, 40, 345, 330, "op", "Strato 2: 10 punteggi", [
    "per ogni cifra c = 0..9:",
    "§acc2[c] = B2[c] + Σ h[j]·W2[j][c]",
    "§          j = 0..15",
    "",
    "16 prodotti h × peso:",
    "  8 bit × 8 bit con segno → 16 bit",
    "accumulatore: 18 bit con segno",
    "",
    "in hardware: si riusano i MAC 0..9,",
    "1 valore h per ciclo → 16 cicli",
]))
f2.append(box(1025, 390, 345, 120, "par", "ROM W2 + bias B2", [
    "16 righe × 10 pesi da 8 bit (80 bit)",
    "riga j = pesi di h[j] per le 10 cifre",
    "§B2 = 96, 71, 382, … (10 valori)",
]))
f2.append(arrow(1197, 390, 1197, 375))

# Valori dell'esempio: acc1 e h come barre
def barre(x0, y0, w, hgt, valori, etichetta, colore, vmax, mostra_val=True, evid=None,
          zero_mid=True):
    s = [text(x0, y0 - 10, etichetta, 14, TXT, "700")]
    n = len(valori)
    bw = w / n
    base = y0 + hgt / 2 if zero_mid else y0 + hgt
    s.append(f'<line x1="{x0}" y1="{base}" x2="{x0 + w}" y2="{base}" stroke="#999" stroke-width="1"/>')
    scala = (hgt / 2 if zero_mid else hgt) / vmax
    for i, v in enumerate(valori):
        bh = abs(v) * scala
        by = base - bh if v >= 0 else base
        col = colore if (evid is None or i == evid) else "#b9c4d3"
        if v < 0:
            col = "#d9a7a7"
        s.append(f'<rect x="{x0 + i * bw + 2}" y="{by}" width="{bw - 4}" height="{max(bh, 0.8)}" '
                 f'fill="{col}"/>')
        s.append(f'<text x="{x0 + i * bw + bw / 2}" y="{y0 + hgt + 26}" font-size="10.5" '
                 f'text-anchor="middle" fill="{SUB}">{i}</text>')
        if mostra_val:
            ty = by - 4 if v >= 0 else by + bh + 12
            s.append(f'<text x="{x0 + i * bw + bw / 2}" y="{ty}" font-size="9.5" '
                     f'text-anchor="middle" fill="{TXT}">{v}</text>')
    return "\n".join(s)

f2.append(text(40, 562, "Esempio numerico: l'immagine 0 attraversa la rete (valori veri del golden model)",
               16, C["dati"][1], "700"))
f2.append(barre(40, 600, 560, 130, acc1, "acc1 (16 neuroni, prima della ReLU)", "#c46a12", 8600))
f2.append(arrow(610, 665, 655, 665))
f2.append(barre(665, 600, 330, 130, h, "h (dopo ReLU, >> 6, saturazione)", "#2f6db5", 120,
                zero_mid=False))
f2.append(arrow(1005, 665, 1035, 665))
f2.append(barre(1045, 600, 325, 130, acc2, "acc2 (punteggi delle 10 cifre)", "#2e8a46", 8600,
                evid=2))
f2.append(text(1045, 778, "il massimo è la cifra 2 (3377) → risposta corretta", 13, C["par"][1], "700"))
f2.append(text(40, 778, "7 neuroni su 16 sono negativi e la ReLU li spegne", 13, SUB))

# Striscia dei bit
y = 822
f2.append(text(40, y, "Larghezza dei dati lungo il percorso", 15, TXT, "700"))
passi = [("pixel", "4 bit u", "dati"), ("× peso", "8 bit s", "par"), ("prodotto", "12 bit s", "op"),
         ("Σ 64 + B1", "18 bit s", "op"), ("ReLU >>6 sat", "", "op"), ("h", "8 bit u", "dati"),
         ("× peso", "8 bit s", "par"), ("prodotto", "16 bit s", "op"), ("Σ 16 + B2", "18 bit s", "op"),
         ("argmax", "", "op"), ("cifra", "4 bit u", "dati")]
bx, bwid, gap = 40, 102, 20
for i, (nome, bit, kind) in enumerate(passi):
    x = bx + i * (bwid + gap)
    fill, stroke = C[kind]
    f2.append(f'<rect x="{x}" y="{y + 12}" width="{bwid}" height="52" rx="8" fill="{fill}" '
              f'stroke="{stroke}" stroke-width="1.6"/>')
    f2.append(text(x + bwid / 2, y + 34, nome, 12.5, TXT, "700", "middle"))
    if bit:
        f2.append(text(x + bwid / 2, y + 54, bit, 12.5, stroke, "400", "middle", mono=True))
    if i < len(passi) - 1:
        f2.append(arrow(x + bwid + 1, y + 38, x + bwid + gap - 1, y + 38))
f2.append(text(40, y + 92, "u = senza segno, s = con segno (complemento a 2). Le scale (2^10, 2^5, 512) "
               "esistono solo in Python: in hardware ci sono solo interi, e il massimo non cambia.",
               12.5, SUB))
f2.append("</svg>")

html = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<title>Schema del modello</title>
<style>
  @page {{ size: A4 landscape; margin: 9mm; }}
  body {{ margin: 0; background: #fff; color: {TXT};
         font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }}
  .pagina {{ page-break-after: always; padding: 4px 8px; }}
  .pagina:last-child {{ page-break-after: auto; }}
  h1 {{ font-size: 21px; margin: 2px 0 2px; }}
  p.sotto {{ margin: 0 0 6px; color: {SUB}; font-size: 13px; }}
  svg {{ display: block; max-width: 100%; max-height: 168mm; width: auto; height: auto; margin: 0 auto; }}
</style></head><body>
<div class="pagina">
  <h1>1. Dalla rete addestrata ai file per il VHDL</h1>
  <p class="sotto">Il percorso in Python (cartella <code>python/</code>): ogni blocco è una funzione o uno script, con le scelte fatte e i numeri misurati.</p>
  {"".join(f1)}
</div>
<div class="pagina">
  <h1>2. Il calcolo della rete a interi (golden model = ciò che farà la FPGA)</h1>
  <p class="sotto">Rete 64 → 16 → 10, pesi a 8 bit, uscite nascoste a 8 bit, accumulatori a 18 bit. Sotto, i valori veri dell'immagine di test n. 0.</p>
  {"".join(f2)}
</div>
</body></html>"""
(OUT / "schema_modello.html").write_text(html)
for nome, f in [("schema_pipeline.svg", f1), ("schema_calcolo.svg", f2)]:
    (OUT / nome).write_text("\n".join(f))
print("ok")
