# Riassunto Completo del Progetto Aesur_ext (13 - 18 Ottobre 2026)

Documento di sintesi dettagliata delle attività, modelli sviluppati, test empirici condotti, risultati numerici verificati su test set e analisi matematiche dello spazio latente.

---

## 1. Obiettivo del Progetto e Contesto Fisico

Creazione di modelli surrogati veloci e compatti basati su **Autoencoder Convoluzionali 1D (CNN)** per comprimere e ricostruire forme d'onda gravitazionali emesse dalla fusione di buchi neri binari (BBH).

- **Dataset**: `NRHybSur3dq8_dataset.hdf` (102.400 forme d'onda, 2048 campioni temporali a frequenza costante).
- **Rappresentazione del segnale**: Decomposizione del modo dominante $(2,2)$ in modulo e fase srotolata:
  $$h(t) = A(t) e^{-i \phi(t)}$$
  - **Ampiezza**: $A(t) = |h(t)| \in [0, \sim 0.45]$ (curva a campana liscia, picco al merger).
  - **Fase**: $\phi(t) = -\frac{\text{unwrap}(\text{angle}(h(t)))}{100}$, con shift $\phi(0) = 0$ (crescita monotonica non lineare da $0$ a oltre $60$ radianti).
- **Dimensioni del Bottleneck**: $z = 64$ variabili latenti scalari.
- **Obiettivo prestazionale**: Portare la loss di ricostruzione MAE nell'ordine di $10^{-5}$ sia su ampiezza che su fase.
- **Benchmark di riferimento**: Il paper *AESur*, che impiega reti dense (Fully Connected / MLP) da 3 a 5.6 milioni di parametri. L'obiettivo è superare tale accuratezza con modelli convoluzionali 1D compresi tra 120k e 260k parametri.

---

## 2. Flusso di Lavoro ed Evoluzione Architetturale

### Fase 1: Setup dell'Ambiente e Dataloading (13 - 14 Ottobre)
1. **Gestione ambiente con `uv`**: Ambiente isolato per riproducibilità su macchine locali e remote.
2. **Dataloading a blocchi da HDF5**: Lettura in chunk (`chunk_size = 10000`) per calcolare ampiezza e fase vettorizzate senza saturare la RAM di sistema.
3. **Split deterministico**: 80% Train (81.920 onde), 10% Validation (10.240 onde), 10% Test (10.240 onde) con seed fisso (`seed=42`).

### Fase 2: Sviluppo delle Architetture CNN 1D
1. **Modello Singolo (`Autoencoder_CNN_1`)**:
   - **Encoder 1**: 3 blocchi `Conv1d` con kernel grande $k=40$ (stridi 2, 4, 4; filtri 32, 16, 8) + `BatchNorm1d` + attivazione, terminante con `Linear(512 -> 64)`.
   - **Decoder 1**: `Linear(64 -> 256)` + `BatchNorm1d` + 3 stadi `ConvTranspose1d` (stridi 2, 2, 2; filtri 32, 32, 8; $k=40$) + `Conv1d(k=7)` finale a 1 canale.
   - **Parametri**: **129.336** (~59.7k encoder, ~69.6k decoder).
   - Due istanze parallele: una dedicata alla sola ampiezza, una alla sola fase.
2. **Modello Congiunto a 2 Canali (`Autoencoder_CNN_2` - DoubleAE)**:
   - Prende in input e produce in output un tensore a 2 canali `(B, 2, 2048)`.
   - **Encoder 2**: `Conv1d` a 2 canali d'ingresso (filtri 64, 32, 16; $k=40$) + `Linear(1024 -> 64)`.
   - **Decoder 2**: `Linear(64 -> 512)` rimodellato in `(B, 2, 256)`, 3 stadi `ConvTranspose1d` e `Conv1d` finale a 2 canali.
   - **Parametri**: **261.088** (~173k encoder, ~87.8k decoder).
3. **Evoluzione delle Attivazioni**:
   - Versione iniziale: `LeakyReLU(0.2)`.
   - Versione aggiornata: `GELU()` su tutti i layer (migliore stabilità di gradiente ed eliminazione di cuspidi artificiali).

### Fase 3: La Crisi del Weight Decay (15 - 16 Ottobre)
- **Problema**: Le prime sessioni da 3000 epoche su HTCondor mostravano un plateaux totale sulla fase ($L_1 > 10^{-3}$).
- **Causa individuata**: Optuna aveva suggerito `weight_decay = 0.0098` sulla fase. Con learning rate $\eta \approx 7 \times 10^{-3}$ su $3.8 \times 10^6$ step totali, il fattore passivo di decadimento $(1 - \eta \lambda)^T$ abbatteva i pesi a zero: la matrice `Linear(512 -> 64)` aveva pesi collassati a $< 10^{-3}$.
- **Risoluzione formale (`weight_decay_rules.ipynb`)**:
  - Regola 1 (Budget di step): $\lambda \approx S / (\eta \cdot T)$ con shrinkage max $10\%$.
  - Regola 2 (Bayesian Dataset Scaling): scaling con $1/N$.
  - Il `weight_decay` è stato abbattuto a $10^{-9}$ o $1.4 \times 10^{-6}$, ripristinando il pieno apprendimento.

### Fase 4: Allineamento con AESur e Regolarizzazione Latente $L_2$ (17 Ottobre)
- Introduzione del ritorno esplicito dello spazio latente (`return_latent=True`).
- Loss combinata:
  $$\mathcal{L}_{\text{tot}} = \text{MAE}(\hat{x}, x) + \lambda_{\text{latent}} \frac{1}{z}\sum_{i=1}^{z} z_i^2 \quad \text{con } \lambda_{\text{latent}} = 10^{-5}$$
- La penalità $L_2$ mantiene i vettori latenti ordinati e previene derive di magnitudine durante lunghe run.

### Fase 5: Dinamica di `ReduceLROnPlateau` (18 Ottobre)
- Chiarito che `patience` conta le epoche perché `scheduler.step(vloss)` viene chiamato a fine epoca.
- Configurazione calibrata: `factor=0.5`, `patience=75-100`, `threshold=0.0001`, `min_lr=1e-6` per evitare drop prematuri che congelavano il modello.

---

## 3. Tutti i Test Condotti e Risultati Numerici

### Test 1: Overfitting a Singolo Batch (64 forme d'onda, 500 step, no regolarizzazione)
Verifica della pura capacità di memorizzazione dell'architettura a parità di dati:

| Modello | Parametri | Target | $L_1$ Loss Finale | Esito |
|---|---|---|---|---|
| **PCA Lineare ($k=64$)** | Analitico | Ampiezza | **$3.007 \times 10^{-8}$** | Limite teorico lineare |
| **PCA Lineare ($k=64$)** | Analitico | Fase | **$3.691 \times 10^{-7}$** | Limite teorico lineare |
| **Autoencoder_CNN_1** (GELU) | ~129k | Ampiezza | **$2.628 \times 10^{-3}$** | Supera il modello denso del paper |
| **Autoencoder_CNN_1** (GELU) | ~129k | Fase | **$1.453 \times 10^{-2}$** | Supera il modello denso del paper |
| **Autoencoder_CNN_2** (Double) | ~261k | 2 Canali | **$1.124 \times 10^{-2}$** | Apprendimento congiunto più rigido |
| **DenseAE** (AESur Paper) | ~3.0M | Ampiezza | **$4.490 \times 10^{-3}$** | Sotto-performante rispetto alla CNN |
| **DenseAE** (AESur Paper) | ~3.0M | Fase | **$3.052 \times 10^{-2}$** | Peggior risultato sulla fase |

### Test 2: Rango Intrinseco del Dataset (Teorema di Eckart-Young via SVD)
Calcolo analitico dell'errore minimo teorico insuperabile con $z=64$:
- **Ampiezza**: per raggiungere $MAE = 10^{-5}$ bastano **7 componenti principali**. Con $z=64$, il limite teorico scende a $9.96 \times 10^{-8}$.
- **Fase**: per raggiungere $MAE = 10^{-5}$ bastano **5 componenti principali**. Con $z=64$, il limite teorico scende a $7.44 \times 10^{-7}$.
- **Conclusione**: il bottleneck $z=64$ ha capacità teorica più che sufficiente; il limite a convergere non è la dimensione latente.

### Test 3: Valutazione Reale su Test Set Indipendente (10.240 forme d'onda)
Metriche calcolate sui checkpoint salvati addestrati tra 1000 e 3000 epoche:

| Checkpoint | Target | MAE Medio | MAE Mediano | Max MAE | Errore Relativo $L_1$ | MSE |
|---|---|---|---|---|---|---|
| `model1_ampl_gelu.pt` | Ampiezza | **$6.268 \times 10^{-5}$** | $5.598 \times 10^{-5}$ | $4.935 \times 10^{-4}$ | **$0.078\%$** ($7.88 \times 10^{-4}$) | $1.41 \times 10^{-8}$ |
| `model2_phase_gelu.pt` | Fase | **$6.281 \times 10^{-4}$** | $5.424 \times 10^{-4}$ | $8.021 \times 10^{-3}$ | **$0.066\%$** ($6.65 \times 10^{-4}$) | $1.76 \times 10^{-6}$ |
| `model_doubleAE_gelu_aesurliketraining.pt` | 2 Canali | **$2.326 \times 10^{-1}$** | $2.140 \times 10^{-1}$ | $0.852$ | $> 25\%$ | $7.84 \times 10^{-2}$ |

*Dettaglio sulla Fase*: un MAE di $6.281 \times 10^{-4}$ sul segnale normalizzato corrisponde a un errore medio reale di **$0.0628$ radianti** ($\approx 3.6^\circ$) lungo l'intera orbita (che copre oltre 60 radianti totali).

### Test 4: Profilo Temporale dell'Errore (Inspiral vs Merger)
Analisi lungo i 2048 campioni temporali (`temporal_mae_ampl_gelu.npy` e `temporal_mae_phase_gelu.npy`):

| Segnale | Inspiral (Primi 1800 campioni) | Merger & Ringdown (Ultimi 248 campioni) | Rapporto Errore |
|---|---|---|---|
| **Ampiezza** | **$5.340 \times 10^{-5}$** | **$1.300 \times 10^{-4}$** | $2.4\times$ |
| **Fase** | **$5.563 \times 10^{-4}$** | **$1.149 \times 10^{-3}$** | $2.1\times$ |

L'errore è uniformemente distribuito su tutta l'ispiralizzazione e raddoppia in corrispondenza del picco del merger. Nessuna oscillazione ad alta frequenza o artefatto spurio è presente.

### Test 5: Analisi dello Spazio Latente $Z \in \mathbb{R}^{10240 \times 64}$
Ispezione della varianza e dei valori singolari di $Z$:
- **Dimensioni attive**: **64 su 64 attive** (nessun collasso dimensionale).
- **Varianza Spiegata (PCA su $Z$)**:
  - Ampiezza: **7 componenti latenti** spiegano oltre il $99\%$ della varianza totale.
  - Fase: **8 componenti latenti** spiegano oltre il $99\%$ della varianza totale.

### Test 6: Condizionamento Spettrale (Neural Tangent Kernel - NTK)
Misura del rapporto tra autovalore massimo e minimo del gradiente:
- CNN 1D: $\kappa \approx 4.6 \times 10^{18}$.
- La loss media privilegia fortemente la componente a bassa frequenza dell'inspiral, assegnando gradienti minuscoli alla transizione rapida del merger.

---

## 4. Matrice di Sintesi: Successi vs Fallimenti

| Approccio Provato | Esito | Dettaglio Tecnico |
|---|---|---|
| **CNN 1D vs Dense (AESur)** | **SUCCESSO** | La CNN 1D (~129k par) batte il DenseAE (3M-5.6M par) con un 23x di risparmio parametrico. |
| **Target Ampiezza ($10^{-5}$)** | **SUCCESSO** | Raggiunto: $MAE = 6.26 \times 10^{-5}$, errore relativo $0.078\%$. |
| **Abbattimento Weight Decay ($10^{-9}$)** | **SUCCESSO** | Ha impedito la cancellazione passiva dei pesi, sbloccando l'addestramento. |
| **GELU + Regolarizzazione Latente $L_2$ ($10^{-5}$)** | **SUCCESSO** | Spazio latente a varianza piena (64/64 dimensioni attive) e training stabile. |
| **Dimensione Latente $z=64$** | **SUCCESSO** | La SVD prova che il rango intrinseco del dataset è $< 10$: $z=64$ non rappresenta un collo di bottiglia limitante. |
| **Weight Decay di Optuna ($0.0098$)** | **FALLITO** | Azzeramento dei pesi pre-latenti e stallo a $L_1 > 10^{-3}$. |
| **DoubleAE 2 Canali (`Autoencoder_CNN_2`)** | **FALLITO** | Interferenza cross-canale nello stesso bottleneck: MAE $0.23$ (collasso rispetto al modello singolo). |
| **Scheduler troppo aggressivo (`factor=0.1`, `patience=250`)** | **FALLITO** | Precipita l'LR a $10^{-6}$ prematuramente o resta bloccato sui plateaux per centinaia di epoche. |

---

## 5. Sintesi Fisica e Prossimi Step Operativi

1. **Abbandonare l'architettura congiunta a 2 canali**: concentrare la computazione sui modelli a canale singolo (`Autoencoder_CNN_1`), nettamente superiori.
2. **Colmare il gap della fase ($6.28 \times 10^{-4} \to 10^{-5}$)**:
   - Poiché l'errore della fase è concentrato al merger ($1.15 \times 10^{-3}$ vs $5.56 \times 10^{-4}$), implementare una **Time-Weighted Loss**:
     $$\mathcal{L}_{\text{weighted}} = \frac{1}{T} \sum_{t=1}^{T} w(t) |\hat{\phi}(t) - \phi(t)| \quad \text{con } w(t) \in [1, 5] \text{ negli ultimi 200 campioni}$$
3. **Mantenere `ReduceLROnPlateau` Calibrato**: Utilizzare la configurazione testata (`factor=0.5`, `patience=75-100`, `threshold=1e-4`, `min_lr=1e-6`) che evita i drop prematuri a $10^{-6}$ e si adatta alla convergenza effettiva.
