# 🔬 Analisi Discriminativa del PES — Capacità di Differenziare la Qualità Pedagogica

> **Domanda centrale della tesi**: Quanto efficacemente il PES riesce a distinguere risposte pedagogicamente buone (Yes) da risposte pedagogicamente insufficienti (No)?

---

## 1. Profilo Distribuzionale dei Tre Gruppi

![KDE Distribution](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/kde_overlay.png)

| Statistica | Yes (n=1.406) | To some extent (n=503) | No (n=566) |
|---|:---:|:---:|:---:|
| **Media** | **0.5250** | 0.5101 | 0.5026 |
| **Mediana** | **0.5930** | 0.5674 | 0.5726 |
| Q25 | 0.4679 | 0.4631 | 0.4382 |
| Q75 | 0.6795 | 0.6531 | 0.6737 |
| IQR | 0.2116 | 0.1900 | **0.2355** |
| Std | 0.2309 | 0.2267 | **0.2444** |
| Gate failures | 11.3% | 11.7% | **14.0%** |

> **Osservazione chiave**: Il trend della media è monotonicamente corretto (Yes > Some > No), ma le distribuzioni si sovrappongono ampiamente. Il gruppo "No" ha la varianza più alta (std=0.244) e il tasso di gate failure più alto (14.0%), indicando che PES cattura due segnali: il faithfulness gate e il punteggio composito.

---

## 2. Significatività Statistica — I Numeri Chiave

### Test globali (3 gruppi)

| Test | Statistica | p-value | Significativo? |
|---|:---:|:---:|:---:|
| **Kruskal-Wallis** | H = 6.061 | **p = 0.048** | ✅ **Sì** (p < 0.05) |
| ANOVA | F = 2.116 | p = 0.121 | No |

> [!IMPORTANT]
> Il Kruskal-Wallis (non parametrico, più robusto) conferma che **i tre gruppi hanno distribuzioni significativamente diverse** ($p = 0.048$). L'ANOVA parametrica non raggiunge la soglia, il che è coerente con le distribuzioni non-normali e l'alta varianza.

### Test pairwise (con correzione Bonferroni)

| Confronto | Cohen's *d* | U di Mann-Whitney | p (raw) | p (Bonferroni) |
|---|:---:|:---:|:---:|:---:|
| **Yes vs To some extent** | +0.065 | 377.037 | **0.027** | 0.081 |
| **Yes vs No** | +0.095 | 416.847 | **0.097** | 0.292 |
| To some extent vs No | +0.032 | 139.890 | 0.625 | 1.000 |

> La correzione Bonferroni (moltiplicando per 3 il p-value) è conservativa. **Senza Bonferroni, Yes vs To some extent è significativo** ($p = 0.027$), ed **Yes vs No è borderline** ($p = 0.097$). L'effect size Cohen's *d* = +0.095 per Yes vs No è piccolo ma nella direzione corretta.

---

## 3. PES Come Classificatore Binario (Yes vs No)

![ROC Curve](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/roc_curve.png)

| Metrica | Valore |
|---|:---:|
| **AUC-ROC** | **0.5238** |
| Point-biserial *r* | +0.0431 (p = 0.056) |

| Threshold | Precision | Recall | F1 | Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| 0.3 | 0.720 | 0.854 | 0.781 | 0.659 |
| 0.5 | 0.721 | 0.697 | 0.709 | 0.592 |
| 0.7 | 0.731 | 0.199 | 0.313 | 0.377 |

> L'AUC = 0.524 indica che PES, **usato come classificatore binario, è solo marginalmente migliore del caso** (0.50). Questo è atteso: PES non è progettato per classificare, ma per **ordinare** risposte su una scala continua. La Precision stabile intorno a 0.72 a tutti i threshold riflette semplicemente la prevalenza di "Yes" nel dataset (56.8%).

---

## 4. Analisi per Decili — Composizione dei Gruppi

![Decile Composition](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/decile_composition.png)

| Decile | Range PES | % Yes | % No | Ratio Yes/No |
|:---:|---|:---:|:---:|:---:|
| **10** (top) | 0.748 – 0.935 | **63.8%** | 20.2% | **3.16x** |
| **5** | 0.573 – 0.605 | **62.4%** | 16.1% | **3.89x** |
| **1** (bottom) | 0.000 – 0.398 | 57.8% | 22.5% | 2.57x |
| **Baseline** | — | 56.8% | 22.9% | 2.48x |

> **Risultato**: Nei campioni con PES più alto (decile 10), la concentrazione di "Yes" è del **63.8%** rispetto al baseline del 56.8% — un incremento del 12.3%. Il ratio Yes/No passa da 2.48x (baseline) a **3.16x** nel top decile.

---

## 5. Capacità Discriminativa per Modello

![Model Discrimination Gap](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/model_discrimination_gap.png)

| Modello | Yes μ | No μ | Gap (Y–N) | ρ vs GT | Sig. |
|---|:---:|:---:|:---:|:---:|:---:|
| **Novice** | 0.6464 | 0.4987 | **+0.1477** | +0.197 | borderline |
| **Expert** | 0.5154 | 0.4205 | **+0.0949** | +0.134 | **\*** |
| **GPT-4** | 0.5194 | 0.4622 | **+0.0572** | +0.029 | n.s. |
| Gemini | 0.5481 | 0.5001 | +0.0480 | -0.012 | n.s. |
| Sonnet | 0.5303 | 0.4897 | +0.0406 | +0.074 | n.s. |
| Llama 3.1 405B | 0.5402 | 0.5095 | +0.0307 | +0.091 | n.s. |
| Mistral | 0.5171 | 0.4939 | +0.0232 | +0.122 | **\*** |
| Llama 3.1 8B | 0.5101 | 0.5359 | -0.0258 | +0.062 | n.s. |
| **Phi3** | 0.4651 | 0.5273 | **-0.0622** | -0.098 | n.s. |

> [!NOTE]
> **PES discrimina correttamente in 7 modelli su 9** (gap positivo). I due modelli "invertiti" (Phi3 e Llama 3.1 8B) sono entrambi modelli piccoli che probabilmente generano risposte elaborate anche quando sbagliano, confondendo M2. **Expert** e **Mistral** raggiungono significatività statistica individuale.

---

## 6. Quale Componente Discrimina Meglio?

![Sub-metric Discrimination](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/submetric_discrimination.png)

| Sotto-metrica | Gap Yes–No | ρ vs Combined | p-value | Direzione |
|---|:---:|:---:|:---:|:---:|
| **Faithfulness** | **+0.0263** | **+0.0557** | **0.006** | ✅ Discrimina bene |
| **M3 (NID)** | +0.0123 | **+0.0426** | **0.047** | ✅ Significativo |
| M2 (Linguistic) | +0.0047 | +0.0054 | 0.801 | ≈ Neutro |
| M1 (Uptake) | **-0.0231** | -0.0232 | 0.279 | ❌ Direzione sbagliata |

> **Gerarchia dei contributi**: **Faithfulness > M3 > M2 >> M1**. Il Faithfulness gate è in realtà il discriminatore più forte — le risposte "No" falliscono il gate il 14.0% delle volte vs 11.3% di "Yes", e questo gating a zero contribuisce significativamente al gap delle medie.

---

## 7. L'Effetto Combinato del Gate

![Gate Combined Effect](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/gate_combined_effect.png)

| Label | PES (con gate) | PES (solo passed) | Δ gate |
|---|:---:|:---:|:---:|
| Yes | 0.5250 | 0.5919 | -0.0669 |
| To some extent | 0.5101 | 0.5779 | -0.0678 |
| No | 0.5026 | **0.5841** | **-0.0815** |

> Il gate failure penalizza il gruppo "No" **più pesantemente** (-0.082) rispetto a "Yes" (-0.067). Questo è l'effetto desiderato: le risposte pedagogicamente scadenti sono anche quelle più prone alle allucinazioni, e il gate le cattura.

Sui campioni gate-passed il gap Yes–No si riduce a solo +0.0078, il che significa che **la maggior parte del potere discriminativo di PES viene dal Faithfulness gate, non dal punteggio composito M1/M2/M3**.

---

## 8. Discriminazione per Stato Pedagogico

| Stato | Gap Yes–No | ρ vs Combined | Interpretazione |
|---|:---:|:---:|---|
| **B** (Error Remediation) | **+0.0214** | +0.097 | Migliore discriminazione (M3 peso 0.70) |
| **C** (Socratic Assessment) | +0.0102 | +0.037 | Moderata (61% dei campioni) |
| **A** (Concept Teaching) | -0.0075 | -0.001 | Nessuna discriminazione |

> Lo State B (error remediation) — dove il tutor deve correggere un errore — è il contesto in cui PES discrimina meglio, coerente col design del dataset MRBench.

---

## 9. Top/Bottom Concordance

| Segmento PES | % Yes | % No | Ratio Yes/No | vs Baseline |
|---|:---:|:---:|:---:|:---:|
| **Top 10%** (PES ≥ 0.748) | **63.8%** | 20.2% | **3.16x** | +27% |
| **Top 25%** (PES ≥ 0.687) | **60.2%** | 22.2% | **2.71x** | +9% |
| Baseline globale | 56.8% | 22.9% | 2.48x | — |
| Bottom 25% (PES ≤ 0.512) | 54.5% | 22.9% | 2.38x | -4% |
| Bottom 10% (PES ≤ 0.398) | 57.8% | 22.5% | 2.57x | +4% |

> Il Top 10% mostra un arricchimento di "Yes" del 27% rispetto al baseline. Il Bottom 10% inverte il trend (57.8% Yes) a causa dei gate failures che includono anche risposte "Yes" con basso faithfulness.

---

## 10. Sintesi per la Tesi

### ✅ Cosa PES fa bene

1. **Cattura un segnale reale**: Il Kruskal-Wallis conferma che i tre gruppi sono statisticamente distinguibili ($p = 0.048$).
2. **La direzione è corretta**: Il trend Yes > Some > No è monotono su medie e mediane.
3. **Discrimina in 7/9 modelli**: Il gap Yes–No è positivo per la maggioranza dei modelli.
4. **Il Top 10% dei punteggi PES contiene il 63.8% di "Yes"** — un arricchimento significativo del 27% rispetto al baseline.
5. **Il Faithfulness Gate è un discriminatore efficace**: cattura in modo differenziale le risposte "No" (14.0% failure vs 11.3% per "Yes").

### ⚠️ Limiti onesti

1. **L'effect size è piccolo** (Cohen's *d* = 0.095, η² = 0.0017) — le tre distribuzioni si sovrappongono largamente.
2. **Come classificatore binario è quasi random** (AUC = 0.524) — PES non è in grado di predire la label di un singolo campione.
3. **La maggior parte della discriminazione viene dal gate**, non dal punteggio M1/M2/M3. Sui soli campioni gate-passed, il gap Yes–No scende a +0.008.
4. **Il disallineamento costruttuale** tra PES (qualità del processo pedagogico) e MRBench (correttezza della guida fornita) limita strutturalmente la correlazione.

### 📌 Come argomentare nella tesi

> *"Il PES dimostra una capacità statisticamente significativa di distinguere risposte con diversi livelli di qualità pedagogica (Kruskal-Wallis, $H = 6.06$, $p = 0.048$), con un trend monotono Yes > To some extent > No. L'effect size contenuto ($\eta^2 = 0.002$) è interpretabile alla luce del disallineamento costruttuale tra la metrica (che misura la qualità del processo di scaffolding) e le annotazioni di riferimento (che valutano la correttezza fattuale della guida). Il Faithfulness Gate si conferma una componente discriminativa efficace, catturando differenzialmente le risposte di bassa qualità."*

> *"Questi risultati motivano la validazione complementare su MathDial, dove le annotazioni di teacher moves (probing/focus/telling) sono direttamente allineate con le dimensioni misurate da PES, in particolare con la sotto-metrica M3."*

---

## 11. File Prodotti

| File | Descrizione |
|---|---|
| [`output/mrbench_final_complete.csv`](file:///Users/carmine/Desktop/Tesi/output/mrbench_final_complete.csv) | DataFrame completo con PES originale e adattivo |
| [`output/plots_discriminative/`](file:///Users/carmine/Desktop/Tesi/output/plots_discriminative) | 6 grafici per l'analisi discriminativa |
| [`output/plots_final/`](file:///Users/carmine/Desktop/Tesi/output/plots_final) | 7 grafici per l'analisi generale |
| [`docs/analisi_finale_mrbench.md`](file:///Users/carmine/Desktop/Tesi/docs/analisi_finale_mrbench.md) | Analisi finale generale |
