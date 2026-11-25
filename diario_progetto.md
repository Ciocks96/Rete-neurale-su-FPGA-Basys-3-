# Diario di progetto – Rete neurale su FPGA (Basys 3)

## 2026-11-24 – presenti: Giovanni

Oggi abbiamo preparato la cartella di lavoro e iniziato a esplorare il dataset `digits`. Ho creato `python/esplora_dati.py` per controllare dimensioni, pixel e numero di immagini per cifra, e per visualizzare qualche esempio.

Il dataset contiene 1797 immagini da 8×8 pixel. `d.images` conserva la griglia, mentre `d.data` contiene gli stessi valori disposti su una riga da 64: è quest'ultima forma che userà la rete. L'ordine è riga per riga, quindi il pixel in posizione (r, c) diventa l'ingresso `r·8 + c`. Dovremo mantenere lo stesso ordine quando prepareremo i dati da inviare alla scheda e la ROM dei pesi.

Altri dati emersi dall'esplorazione:
- pixel interi da 0 a 16 (memorizzati come float64).
- le cifre sono distribuite in modo abbastanza uniforme: da 174 a 183 immagini ciascuna. Useremo una divisione stratificata per mantenere questa proporzione anche tra addestramento e test.
- il 48,9% dei pixel vale 0. Tre posizioni (0, 32 e 39) sono sempre vuote; altre nove, tutte sul bordo, sono accese in meno del 2% delle immagini.
- i pixel a 16 sono 10.456 su 115.008 (circa 9%).

La scelta sui pixel è di usare 4 bit e saturare il valore 16 a 15. Con 4 bit si arriva a 15; rappresentare anche 16 richiederebbe 5 bit per un solo valore. La perdita sembra minima (15 al posto del pixel più scuro) e si risparmiano bit nel moltiplicatore e nella memoria. La saturazione è applicata in Python quando carichiamo i dati (`PIXEL_MAX = 15` in `modello.py`) e andrà applicata anche alle immagini inviate alla scheda. Non bisogna semplicemente lasciare che il 16 vada in overflow: diventerebbe 0.

Per ora teniamo tutti i 64 ingressi, anche se tre pixel sono sempre vuoti: eliminarli farebbe risparmiare qualche peso e ciclo, ma complicherebbe la corrispondenza tra pixel e ingressi. La sparsità (quasi metà dei prodotti peso × pixel vale zero) è un'osservazione utile, ma con un MAC sequenziale non ci dà un vantaggio immediato. Resta da confrontare l'accuratezza con pixel originali e saturati, cosa che faremo al passo 4.

Ho avuto qualche problema iniziale con l'ambiente. VS Code usava un'installazione di Python senza le librerie, mentre iCloud rimuoveva alcuni file dell'ambiente dalla Scrivania. Ho spostato l'ambiente in `.venv.nosync`, fuori dalla sincronizzazione. Per le sottolineature di Pylance su `d.images` è bastato specificare il tipo con `cast(Bunch, load_digits())`; il cast non cambia il risultato a runtime. Infine, Code Runner restava in attesa perché il comando usava `$workspaceRoot` senza il nome dello script: ho impostato `$fullFileName`.

Da ricordare: con 4 bit senza segno i valori vanno da 0 a 15; se un numero supera il massimo, l'overflow può farlo ripartire da 0. Per evitarlo servono più bit oppure una saturazione.

### Passo 2 – caricamento e divisione dei dati (`python/modello.py`)

Abbiamo aggiunto a `modello.py` `carica_dati()`, che converte i pixel in interi, li satura a 15 e divide il dataset in modo stratificato. Il modulo contiene anche le costanti comuni (`PIXEL_MAX`, `SEME` e `FRAZIONE_TEST`) e un controllo che parte solo se si esegue direttamente il file.

La divisione produce 1347 immagini per l'addestramento e 450 per il test. Ogni cifra ha tra il 24,7% e il 25,3% dei propri esempi nel test. Richiamando `carica_dati()` due volte si ottiene sempre la stessa divisione.

Per ora teniamo le funzioni condivise in un solo modulo e gli script separati per fase: così non duplichiamo il codice, ma nemmeno mescoliamo tutte le prove in un unico file. Applichiamo la saturazione prima della divisione: è una trasformazione pixel per pixel e non dipende dai dati. Se in futuro calcoleremo parametri dai dati (per esempio una media per normalizzare), li ricaveremo solo dal gruppo di addestramento. Il test resta da parte per la valutazione, non per scegliere le impostazioni.

### Passo 3 – classificatore lineare di riferimento (`python/addestra.py`)

Prima della rete abbiamo provato `LogisticRegression`, che assegna un punteggio a ciascuna cifra e sceglie quello più alto. Ci serviva un riferimento semplice per capire quanto avrebbe aggiunto la rete neurale.

Numeri misurati:
- accuratezza su addestramento **100,00%**, su test **96,44%** (16 errori su 450);
- 10 × 64 = **640 pesi** + 10 bias, **640 moltiplicazioni** per immagine;
- confusione più frequente: **8 scambiato per 1** (3 volte); la colonna "1" raccoglie errori da 2, 4, 6, 8.
- confronto con prove precedenti senza saturazione: 96,00% (432/450) contro 96,44% (434/450) con saturazione a 15. Differenza di 2 immagini (rumore), ma nessuna perdita: primo indizio a favore della decisione sui pixel a 4 bit (verifica vera al passo 4).

La differenza tra addestramento e test è contenuta. Gli errori più comuni coinvolgono la cifra 1; a questa risoluzione un 8 stretto può somigliare a un tratto verticale. In hardware, 640 cicli a 100 MHz richiederebbero circa 6,4 µs, molto meno dei circa 5,6 ms necessari per ricevere un'immagine via seriale.


### Passo 4 – rete neurale in virgola mobile (`python/addestra.py`)

Abbiamo aggiunto la rete 64 → 16 (ReLU) → 10 con `MLPClassifier`. Gli ingressi vengono divisi per 16 (una potenza di 2, quindi in hardware basta spostare la virgola); `carica_dati()` permette anche di usare i pixel originali 0–16, così possiamo confrontarli con quelli saturati. Per non basarci su una sola inizializzazione, abbiamo ripetuto l'addestramento con cinque semi.

Numeri misurati:
| | Test | Addestramento | Pesi |
|---|---|---|---|
| lineare (64 → 10) | 96,44% (16 errori) | 100,00% | 640 |
| rete 64 → 16 → 10, seme 0 | 97,11% (13 errori) | 99,93% | 1184 (1,9 volte) |
| rete, media su 5 semi, pixel 0–15 | **97,29%** (min 97,11, max 97,56, dev. std 0,17) | | |
| rete, media su 5 semi, pixel 0–16 | 97,16% (min 96,67, max 97,78, dev. std 0,38) | | |

Al variare del seme, la rete va dal 97,11% al 97,56%; il seme 0, usato nella prima prova, è tra quelli che hanno fatto peggio. La media è comunque circa 0,85 punti sopra il lineare, con quasi il doppio dei pesi. Tra pixel saturati e originali la differenza media è solo 0,13 punti, meno di un'immagine su 450 e dentro la variabilità osservata. La saturazione a 15 non sembra quindi penalizzare il risultato. Rimane anche qualche errore 8 → 1.

Il test contiene solo 450 immagini, quindi differenze inferiori a un punto vanno prese con cautela. Inoltre non useremo il test per scegliere il seme: la divisione dei dati resta fissa (`SEME`), mentre `random_state` cambia l'inizializzazione della rete. Per confrontare configurazioni useremo più semi e guarderemo la media; per la rete esportata ne fisseremo uno, così da poter riprodurre i risultati.

Il confronto conferma che non serve una rete grande: il lineare raggiunge già il 96,44%, e la rete guadagna circa un punto. Per il progetto teniamo la rete, anche perché il lavoro riguarda il porting su hardware e un'architettura a due strati. In `addestra.py` restano le prove utili a queste scelte; il risultato del lineare lo annotiamo qui come riferimento.



### Passo 5 – quanti neuroni nascosti? (`python/addestra.py`)

Abbiamo confrontato 4, 8, 16, 32 e 64 neuroni nascosti, usando cinque semi per ciascuna dimensione. Oltre all'accuratezza abbiamo stimato memoria e tempo con un solo moltiplicatore. Il grafico è in `python/grafici/addestra_1.png`.

Notazione: "rete 64-16-10" = 64 ingressi (i pixel), 16 neuroni nascosti (ReLU), 10 uscite (le cifre). Pesi = 64 × 16 + 16 × 10 = 1184 (+ 26 bias).

Numeri misurati:
| Neuroni | Accuratezza media | Min – max | Pesi (ROM a 8 bit) | Cicli (stima) | Tempo a 100 MHz |
|---|---|---|---|---|---|
| 4 | 89,60% | 83,56 – 93,78 | 296 (0,29 KB) | 324 | 3,2 µs |
| 8 | 96,09% | 95,56 – 96,44 | 592 (0,58 KB) | 628 | 6,3 µs |
| **16** | **97,29%** | 97,11 – 97,56 | **1184 (1,16 KB)** | **1236** | **12,4 µs** |
| 32 | 97,24% | 96,00 – 98,00 | 2368 (2,31 KB) | 2452 | 24,5 µs |
| 64 | 97,64% | 97,56 – 97,78 | 4736 (4,62 KB) | 4884 | 48,8 µs |

La stima dei cicli conta un ciclo per il bias, uno per ogni ingresso e uno per salvare ciascun neurone: per la rete scelta, `16 × 66 + 10 × 18`. Il conteggio esatto dipenderà dal VHDL.

Con 4 neuroni l'accuratezza è bassa e varia molto tra i semi. Passando da 8 a 16 sale di circa 1,2 punti; oltre 16 resta intorno al 97%, mentre memoria e cicli raddoppiano a ogni aumento. Anche i 32 neuroni fanno leggermente peggio dei 16, una differenza che sembra dovuta alla variabilità.

Abbiamo scelto 16 neuroni. È il modello più piccolo tra quelli provati che raggiunge l'accuratezza massima senza aumentare inutilmente memoria e tempo: circa 1,2 KB di pesi e 12 µs per immagine, contro 5,6 ms per riceverla via seriale. Il fatto che 16 sia una potenza di due semplifica anche l'indirizzamento dei pesi. Gli 8 neuroni perdono circa 1,2 punti; 32 e 64 costano di più senza un miglioramento apprezzabile.

### Passo 6 – il calcolo della rete rifatto "a mano" (`python/quantizza.py`)

Invece di usare `predict()` per ottenere la risposta, abbiamo estratto i pesi e rifatto il calcolo con numpy: somme pesate, ReLU e scelta del punteggio più alto. È lo stesso lavoro che dovrà fare la scheda. Abbiamo anche spostato in `modello.py` la configurazione della rete e `addestra_rete()`, così gli script usano tutti gli stessi valori.

In `quantizza.py` abbiamo controllato il risultato su tutto il test e seguito un'immagine strato per strato. La verifica della divisione per 16 mostra che si può incorporare nei pesi del primo strato e usare i pixel interi.

Per ogni immagine, il calcolo è:
```
z1 = x · W1 + b1        strato 1: 16 somme pesate dei 64 pixel, più il bias
h  = max(0, z1)         ReLU: i negativi diventano 0
z2 = h · W2 + b2        strato 2: 10 somme pesate delle 16 uscite nascoste
cifra = posizione del valore più alto di z2
```

Risultati della verifica:
- W1 (64 × 16) va da −1,68 a 1,68; b1 da −0,25 a 0,59; W2 (16 × 10) da −2,54 a 1,84; b2 da −0,60 a 0,75. Questi intervalli serviranno per scegliere la scala della quantizzazione.
- il nostro calcolo dà **la stessa cifra di `predict()` su 450 immagini su 450** (accuratezza 97,11%, rete con seme 0).
- immagine di test n. 0 (un 2): la ReLU spegne 7 neuroni nascosti su 16; il punteggio più alto è quello della cifra 2 (6,49, il secondo è 3,17).
- dividere i pixel per 16 oppure dividere una volta i pesi W1 per 16 e usare i pixel interi dà **lo stesso risultato** su tutte le immagini: x/16 · W1 = x · (W1/16). La differenza è esattamente 0, perché in binario dividere per 16 sposta solo la virgola e non introduce arrotondamenti.

In hardware useremo pixel interi da 0 a 15: la divisione per 16 è già incorporata nei pesi. Per classificare basta scegliere il punteggio più alto, senza calcolare la softmax.



### Passo 7 – la quantizzazione (`python/modello.py`, `python/quantizza.py`)

In questo passo abbiamo aggiunto a `modello.py` la quantizzazione e `predici_intero()`. Quest'ultima è la versione di riferimento per il VHDL. In `quantizza.py` confrontiamo i risultati in virgola mobile e quelli interi, compreso il calcolo sulla stessa immagine vista al passo precedente.

Il calcolo della rete a interi:
```
acc1 = pixel · W1 + b1                     pixel interi 0..15, pesi interi -127..127
h    = min(255, max(0, acc1) >> SHIFT)     ReLU + shift + saturazione -> 8 bit senza segno
acc2 = h · W2 + b2
cifra = posizione del valore più alto di acc2 (a parità vince la cifra più bassa)
```

Le scelte per la quantizzazione:
- **pesi a 8 bit con segno, quantizzazione simmetrica con scala a potenza di 2 (virgola fissa)**: peso intero = round(peso reale × 2^n), con n il più grande possibile senza che il peso più grande in modulo superi 127 (finisce quindi tra 64 e 127). n è il numero di bit frazionari. Strato 1: i pesi sono prima divisi per 16 (divisione dei pixel assorbita, passo 6). Tutte le scale della rete sono così potenze di 2.
- **bias alla scala dei prodotti**: b1 × scala1; b2 × scala2 × scala1 / 2^SHIFT (perché i bias si sommano ai prodotti nell'accumulatore, devono avere la stessa scala). Moltiplicare tutti i punteggi finali per lo stesso numero positivo non cambia il massimo.
- **uscite nascoste a 8 bit senza segno (0–255)**, ottenute con ReLU + shift + saturazione.
- **SHIFT scelto provando i valori da 0 a 20 sulle immagini di addestramento** (mai sul test): accuratezza più alta, a parità meno uscite sature, poi shift più piccolo.
- confronto con la scala libera (peso più grande portato esattamente a 127): stessa accuratezza sul test (97,11%), quindi la virgola fissa non costa nulla.

Numeri misurati:
- scale: strato 1 = 2^10 = 1024 sui pesi ÷16, cioè pesi reali in formato **Q2.6** (2 bit di parte intera con il segno, 6 frazionari); strato 2 = 2^5 = 32, formato **Q3.5** (i pesi del secondo strato arrivano a valori più grandi).
- interi: W1 da −108 a 108, b1 da −252 a 600, W2 da −81 a 59, b2 da −309 a 382.
- scelta dello shift (accuratezza sull'addestramento / uscite sature):

| SHIFT | 3 | 4 | 5 | **6** | 7–9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| accuratezza | 67,2% | 96,3% | 99,9% | **99,9%** | 99,6–99,9% | 97,8% | 90,7% | 54,0% |
| sature | 44,7% | 15,5% | 0,7% | **0%** | 0% | 0% | 0% | 0% |

  Con uno shift piccolo le uscite tendono a saturarsi a 255; con uno grande si perde precisione. Tra 5 e 9 i risultati sono simili. Abbiamo scelto **SHIFT = 6**, il primo valore senza saturazioni.
- **float contro interi sul test: 97,11% contro 97,11%** (437 giuste in entrambi i casi), stessa risposta su 449 immagini su 450. La quantizzazione a 8 bit non fa perdere nulla.
- uscite nascoste sul test: massimo 169 (sotto 255, nessuna satura).
- accumulatori osservati sul test: acc1 da −4091 a 10854, acc2 da −17523 a 10649 (il caso peggiore teorico si calcolerà al passo 9).
- immagine n. 0: le uscite nascoste intere, riportate alla scala reale (h × 2^6 / scala1), valgono quasi esattamente quelle in virgola mobile (es. 6,75 contro 6,79); i punteggi interi mantengono lo stesso ordine (la cifra 2 vince con 3377).

#### Concetto chiarito – la quantizzazione, perché e come

1. **Perché.** Sulla FPGA i calcoli in virgola mobile costano molti LUT, DSP e cicli; quelli con gli interi sono semplici e veloci. Un moltiplicatore 8×8 bit sta in un solo blocco DSP. Inoltre, con pesi a 8 bit, tutta la rete (1184 pesi) occupa poco più di 1 KB di ROM.
2. **Idea di base: reale ≈ intero / scala.** Ogni peso reale si moltiplica per una scala e si arrotonda. Esempio: 0,0518 × 1024 = 53,0 → intero **53**. L'errore massimo è mezzo passo, cioè 0,5 / 1024 ≈ 0,00049.
3. **Come si sceglie la scala.** Si prende la potenza di 2 più grande che non fa superare 127 (il massimo con 8 bit con segno) al peso più grande in modulo. Una scala più grande farebbe saturare i pesi più grandi; una più piccola sprecherebbe livelli e aumenterebbe l'errore. La scala libera porterebbe il massimo esattamente a 127; con la potenza di 2 il massimo finisce tra 64 e 127 (si può perdere fino a mezzo bit), ma in cambio ogni scala diventa uno spostamento della virgola. Nel nostro caso scala1 = 2^10 e scala2 = 2^5.
4. **Perché la somma funziona, e perché il bias va scalato allo stesso modo.** Se ogni peso è moltiplicato per scala1, anche la somma dei prodotti è moltiplicata per scala1. Il bias si somma a quei prodotti, quindi deve avere la stessa scala: b1 × scala1. Senza questo accorgimento, il bias avrebbe un peso sbagliato. Immagine n. 0, neurone 0: acc1 = **6928**, contro 6,79 × 1024 ≈ 6957 (6,79 è il valore reale calcolato al passo 6).
5. **Riquantizzazione tra gli strati (ReLU + shift + saturazione).** acc1 è troppo grande per entrare nel secondo strato come numero a 8 bit, quindi va riportato tra 0 e 255.
   - ReLU: i negativi diventano 0.
   - Shift: `>> 6` divide per 64, per esempio 6928 >> 6 = **108**. Si usa una potenza di 2 perché in hardware lo shift non costa niente: basta prendere i bit più alti dell'accumulatore, senza bisogno di un divisore.
   - Saturazione: i valori sopra 255 diventano 255, invece di "ripartire da 0" (overflow), che stravolgerebbe il risultato.
   - Il valore dello shift è un compromesso. Se è troppo piccolo, molte uscite si saturano; se è troppo grande, i valori diventano piccoli e si perde precisione (tabella sopra).
6. **Le scale si accumulano, ma non servono.** Dopo lo strato 2 i punteggi sono moltiplicati per scala2 × scala1 / 2^6 = 2^5 × 2^10 / 2^6 = **512**. Per esempio, il punteggio reale della cifra 2 è 6,49, e 6,49 × 512 ≈ 3323, contro i 3377 calcolati a interi. Tutti i punteggi hanno la stessa scala positiva, quindi il più alto resta il più alto: la scheda non deve mai "tornare ai numeri reali". Le scale esistono solo in Python, per costruire i pesi.
7. **Perché non si perde accuratezza.** L'errore di arrotondamento di ogni peso è piccolo e casuale, e nelle somme di molti prodotti gli errori in parte si compensano. Per cambiare la risposta servirebbe un errore più grande della distanza tra il primo e il secondo punteggio, che di solito è ampia.
8. **Tecniche esistenti (e cosa abbiamo scelto).**

   | Scelta | Alternative | Noi |
   |---|---|---|
   | quando si quantizza | dopo l'addestramento (PTQ) / durante l'addestramento (QAT, più preciso con pochi bit) | PTQ |
   | intervallo | simmetrico (zero = 0) / asimmetrico (con offset, *zero point*) | simmetrico |
   | scala | libera / potenza di 2 (virgola fissa, solo shift) | potenza di 2 |
   | granularità | una scala per strato / una per neurone | per strato |
   | arrotondamento | al più vicino / troncamento | al più vicino (in Python); troncamento nello shift |
   | calibrazione | scelta dei parametri su dati di esempio | shift scelto sui dati di addestramento |
   | casi estremi | reti binarie o ternarie (pesi ±1) | da valutare al passo 8 |

9. **Bit in hardware** (calcolati al passo 9):

   | Grandezza | Bit | Perché |
   |---|---|---|
   | pixel | 4, senza segno | 0..15 |
   | pesi | 8, con segno | −127..127 |
   | prodotto pixel × peso | 12, con segno | 15 × 127 = 1905 |
   | accumulatore strato 1 | 16, con segno | 64 prodotti + bias, caso peggiore con i nostri pesi |
   | uscita nascosta h | 8, senza segno | dopo ReLU + shift + saturazione |
   | accumulatore strato 2 | 18, con segno | 16 prodotti h × peso (fino a 255 × 127) + bias, caso peggiore |

#### Concetto chiarito – l'accumulatore

Un neurone deve calcolare una somma di 64 prodotti (strato 1) o di 16 prodotti (strato 2). Con un solo moltiplicatore (un MAC, *multiply-accumulate*) si calcola **un prodotto per ciclo di clock**. Serve quindi un registro che conservi la **somma parziale** da un ciclo all'altro: è l'accumulatore.
```
ciclo 0:   acc <- bias
ciclo 1:   acc <- acc + pixel[0]  · W[0]
ciclo 2:   acc <- acc + pixel[1]  · W[1]
...
ciclo 64:  acc <- acc + pixel[63] · W[63]   -> acc1 completo (es. 6928)
poi:       h = saturazione(ReLU(acc) >> 6)  -> 108, salvato in RAM; acc si azzera per il neurone successivo
```
- **Perché non basta un singolo valore a 8 bit?** Ogni prodotto arriva fino a 1905, e la somma di 64 prodotti fino a circa 122 000. Il risultato finale ci sta in 8 bit solo *dopo* lo shift; durante la somma servono molti più bit.
- **Perché con segno?** I pesi sono negativi o positivi, quindi la somma parziale può scendere sotto zero e poi risalire. La ReLU si applica solo alla fine.
- **Quanto deve essere largo?** Abbastanza per il **caso peggiore**, non solo per quelli osservati (acc1 fino a 10854, acc2 fino a −17523). Se è troppo stretto, in hardware non c'è nessun errore: il numero "gira" (overflow) e si ottiene un valore sbagliato senza accorgersene. Al passo 9 abbiamo calcolato la larghezza esatta: 18 bit.
- **Perché si parte dal bias?** Caricare il bias nell'accumulatore al primo ciclo equivale a sommarlo, senza bisogno di un sommatore in più. Per questo la stima dei cicli è H × (1 + 64 + 1) per lo strato 1: un ciclo per il bias, 64 per i prodotti e uno per salvare il risultato.
- **Versione parallela** (quella scelta, vedi sotto): 16 MAC, uno per neurone, che lavorano insieme, con 16 accumulatori. È circa 14 volte più veloce della versione con un solo MAC, ma usa 16 moltiplicatori.


### Passo 8 – quanti bit servono? (`python/quantizza.py`)

Abbiamo ripetuto la quantizzazione con 8, 6, 5, 4, 3 e 2 bit, sia per i pesi sia per le uscite nascoste h, provando tutte le combinazioni (36 in totale). Lo shift viene scelto ogni volta sui dati di addestramento. Ogni prova è stata ripetuta su 5 reti (semi 0–4), la cui media in virgola mobile è 97,29%. Accuratezza media sul test:

| bit | pesi (h a 8 bit) | h (pesi a 8 bit) | ROM pesi |
|---|---|---|---|
| 8 | 97,38% | 97,38% | 1184 B |
| 6 | 97,20% | 97,20% | 888 B |
| 5 | 96,71% | 97,07% | 740 B |
| 4 | **91,91%** | 96,76% | 592 B |
| 3 | **80,58%** | 95,91% | 444 B |
| 2 | **11,47%** | 93,78% | 296 B |

Cosa ci dicono i risultati:
- **I pesi reggono fino a 5 bit**; da 4 bit in giù l'accuratezza crolla. Il motivo è che la scala è decisa dal peso più grande, quindi con pochi livelli quasi tutti i pesi piccoli diventano 0. Per la rete con seme 0, i pesi a zero del primo strato sono il 9% con 8 bit, il 26% con 4 bit, il 70% con 3 bit e il 95% con 2 bit: la rete "perde" gran parte delle connessioni, e con 2 bit tira a indovinare. Per scendere così in basso servirebbe l'addestramento consapevole della quantizzazione (QAT), che non usiamo.
- **Le uscite nascoste reggono fino a 4 bit** (96,76%). Le h sono meno delicate: sono solo 16 valori per immagine, e lo shift si adatta all'intervallo disponibile.
- Nella griglia completa i due effetti si sommano. Per esempio, pesi a 5 bit e h a 4 bit danno 96,53%.
- Differenze sotto circa 1 punto sono rumore: un'immagine vale 0,22 punti.

**Decisione: restiamo a 8 bit per pesi e h.** Il margine è ampio: si potrebbe scendere a 5 bit per i pesi e 4 per h, ma il risparmio non conta:
- la ROM passa da circa 1,2 KB a 0,7 KB, su 225 KB di Block RAM della Basys 3;
- un blocco DSP moltiplica comunque fino a 25×18 bit, quindi meno bit non fanno risparmiare DSP;
- 8 bit corrispondono a un byte, il che rende semplici l'esportazione e il debug.

Questa prova va comunque citata nella relazione, perché mostra quanto margine c'è.

#### Decisione – architettura parallela

Il punto di forza di una FPGA è poter fare molte operazioni nello stesso ciclo, quindi come versione principale facciamo la **rete parallela con un MAC per neurone nascosto** (16 MAC). Se resta tempo, la versione sequenziale con un solo MAC servirà come confronto di area e velocità per la relazione.

```
strato 1:  per k = 0..63:  pixel[k] va a tutti i 16 MAC
                           MAC j:  acc[j] += pixel[k] · W1[k, j]
           poi:            h[j] = sat(ReLU(acc[j]) >> 6), per tutti e 16 insieme
strato 2:  per j = 0..15:  h[j] va ai MAC 0..9 (riusati)
                           MAC c:  acc[c] += h[j] · W2[j, c]
           poi:            cifra = argmax dei 10 punteggi
```

| | Sequenziale (1 MAC) | **Parallelo (16 MAC)** |
|---|---|---|
| cicli per immagine (stima) | ~1236 (12 µs) | **~90 (meno di 1 µs)** |
| moltiplicatori / accumulatori | 1 / 1 | 16 / 16 |
| parola della ROM dei pesi | 8 bit | 16 × 8 = 128 bit (strato 1), 10 × 8 = 80 bit (strato 2) |
| controllo | due contatori annidati (neurone, ingresso) | un solo contatore (0..63, poi 0..15) |

Conseguenze:
- **Python non cambia.** Le due versioni eseguono gli stessi calcoli interi in ordine diverso, quindi `predici_intero()` resta il riferimento bit per bit.
- **La ROM va organizzata per riga.** All'indirizzo k ci sono i 16 pesi del pixel k, che i 16 MAC leggono nello stesso ciclo. Lo strato 2 si organizza allo stesso modo: all'indirizzo j ci sono i 10 pesi di h[j]. L'esportazione (passo 10) produrrà questo formato.
- Le risorse non sono un problema: 16 moltiplicatori sulla Basys 3 occupano 16 DSP su 90, oppure una parte dei 20 800 LUT se li costruiamo a mano.
- In ogni caso la rete è molto più veloce della seriale (circa 5600 µs per ricevere un'immagine). Il vantaggio del parallelo si misura in risorse e cicli, ed è quello che mostreremo nel confronto.

### Passo 9 – quanti bit servono agli accumulatori? (`python/modello.py`, `python/quantizza.py`)

In `modello.py` abbiamo aggiunto tre funzioni:
- `bit_con_segno()`: quanti bit in complemento a 2 servono per un intervallo di valori;
- `casi_peggiori()`: il valore minimo e massimo che ogni grandezza può assumere con **qualsiasi** immagine in ingresso;
- `tronca_a_bit()`: simula un registro troppo stretto, cioè l'overflow.

Come si calcola il caso peggiore: per ogni neurone, il massimo si ottiene con l'ingresso al massimo (15 per i pixel, 255 per h) dove il peso è positivo e a 0 dove è negativo; il minimo nel caso opposto. Le somme parziali stanno sempre dentro questo intervallo, perché ognuna è il bias più una parte dei prodotti.

| Grandezza | Caso peggiore | Bit | Osservato (tutte le 1797 immagini) | Bit |
|---|---|---|---|---|
| prodotto strato 1 (pixel × peso) | −1905 .. 1905 | 12 | | |
| bias strato 1 | −252 .. 600 | 11 | | |
| **acc1** | −16 333 .. 20 985 | **16** | −4091 .. 11 153 | 15 |
| prodotto strato 2 (h × peso) | −32 385 .. 32 385 | 16 | | |
| bias strato 2 | −309 .. 382 | 10 | | |
| **acc2** | −86 318 .. 48 141 | **18** | −18 596 .. 11 814 | 16 |

Se si riaddestrasse la rete con pesi qualsiasi a 8 bit, il limite sarebbe più alto: 18 bit per acc1 e 20 per acc2. Il caso peggiore con i nostri pesi però è sufficiente, perché i pesi sono costanti nella ROM. L'esportazione ricalcolerà la larghezza ogni volta che cambiano.

Prova pratica (`tronca_a_bit`): stessa rete, accumulatore più stretto, accuratezza sul test.

| Bit dell'accumulatore | 12 | 13 | 14 | 15 | 16 | 17 | **18** |
|---|---|---|---|---|---|---|---|
| accuratezza | 11,1% | 12,9% | 63,1% | 96,9% | 97,1% | 97,1% | **97,1%** |
| immagini con overflow | 450 | 450 | 195 | 1 | 0 | 0 | **0** |

Con 15 bit una sola immagine va in overflow, e l'accuratezza sembra ancora buona: è proprio il caso pericoloso, perché un errore così passa inosservato nelle prove. Con 16 bit il test è tutto corretto, ma non abbiamo la garanzia che basti per *ogni* immagine possibile, che è invece data dal caso peggiore.

Da ricordare: in complemento a 2, troncare solo il risultato finale equivale a troncare ogni somma parziale (l'aritmetica "gira" allo stesso modo). Per questo la simulazione dell'overflow si può fare sull'accumulatore completo.

**Decisione: accumulatore a 18 bit con segno**, uguale per i due strati. Nella versione parallela i MAC 0..9 vengono riusati per lo strato 2, quindi l'accumulatore deve bastare per entrambi: max(16, 18) = 18. Il moltiplicatore condiviso deve gestire 8 bit senza segno (h; i pixel a 4 bit ci stanno) per 8 bit con segno (il peso), con un prodotto a 16 bit.

### Passo 10 – esportazione per il VHDL (`python/esporta.py`)

`esporta.py` riaddestra la rete (seme 0, quindi sempre la stessa), la quantizza e genera due file nella nuova cartella `vhdl/`:

**`vhdl/rete_pkg.vhd`**, un package VHDL con:
- le costanti `N_IN = 64`, `N_HID = 16`, `N_OUT = 10`, `PIXEL_BIT = 4`, `PESO_BIT = 8`, `H_BIT = 8`, `ACC_BIT = 18`, `SHIFT = 6`. `ACC_BIT` viene ricalcolato dal caso peggiore a ogni esportazione;
- `W1`: 64 righe da 128 bit. La riga k contiene i 16 pesi del pixel k, che i 16 MAC leggono nello stesso ciclo;
- `W2`: 16 righe da 80 bit. La riga j contiene i 10 pesi dell'uscita nascosta h[j];
- `B1` e `B2`: i bias come interi, già alla scala degli accumulatori.

Formato di una riga: il peso della colonna j occupa i bit 8j+7 .. 8j (colonna 0 a destra nel valore esadecimale), in complemento a 2. Esempio, riga 1 di W1: `x"E7CF…0835"`, dove `35` (in fondo a destra) = 53 è il peso del neurone 0, `08` = 8 quello del neurone 1, e `E7` (in testa) = −25 quello del neurone 15. Accanto a ogni riga, un commento riporta i valori in decimale per leggerla senza convertire.

Le righe 0, 24, 32 e 39 di W1 sono tutte a zero: sono pixel sempre vuoti, o quasi (passo 1), quindi i loro pesi sono rimasti piccolissimi e arrotondati a 0.

Il file è generato e non va modificato a mano: se cambia la rete, si rilancia `esporta.py`.

**`vhdl/vettori_test.txt`**: le 450 immagini di test, una per riga, con numeri decimali separati da spazi (si leggono facilmente in VHDL con `textio`):
```
64 pixel | 16 h attesi | 10 punteggi acc2 attesi | cifra prevista | cifra vera
```
Il testbench dovrà confrontare h, punteggi e cifra con questi valori: devono coincidere **bit per bit**. Controllare anche h e i punteggi, oltre alla cifra, permette di capire *dove* nasce un errore: nello strato 1 o nello strato 2.

Controllo automatico: lo script rilegge il package, ricostruisce i pesi dalle stringhe esadecimali e ricalcola la rete partendo dai file. Tutto coincide con Python, e l'accuratezza attesa in hardware è **97,11% (437 su 450)**.

Prossimo: il VHDL della rete parallela. Prima i blocchi (MAC con accumulatore a 18 bit, ReLU + shift + saturazione, argmax), poi il controllo con un contatore (64 cicli per lo strato 1, 16 per lo strato 2), poi il testbench che legge `vettori_test.txt`. Da decidere con il gruppo: moltiplicatori nei DSP oppure costruiti a mano nei LUT.