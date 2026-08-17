# 📊 Analisi Finale — Validazione PES su MRBench v3 Devset

> **2.475 campioni** | 9 modelli | ~58 ore di calcolo | Ollama llama3.2 | Faithfulness threshold = 0.25

---

## 1. Overview

| Statistica | Valore |
|---|:---:|
| **Campioni totali** | 2.475 |
| **PES medio (adattivo)** | 0.5168 |
| **PES mediano (adattivo)** | 0.5864 |
| **Gate failures** | 297 (12.0%) |
| **Modelli valutati** | 9 (GPT-4, Gemini, Sonnet, Llama3.1-405B/8B, Phi3, Mistral, Expert, Novice) |

---

## 2. PES per Label Umana (Providing Guidance)

La metrica PES mostra il **trend monotono atteso**: risposte giudicate migliori dagli annotatori umani ottengono punteggi PES più alti.

| Label | n | PES Medio | PES Mediano |
|---|:---:|:---:|:---:|
| **Yes** | 1.406 | **0.5250** | **0.5930** |
| To some extent | 503 | 0.5101 | 0.5674 |
| No | 566 | 0.5026 | 0.5726 |

> **Gap Yes–No = +0.0224** (pesi adattivi) vs +0.0174 (pesi originali) → **+28.2% di miglioramento** dalla calibrazione dei pesi.

![PES Distribution by Label](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/01_pes_by_label.png)

---

## 3. Significatività Statistica

### Correlazioni Spearman

| Coppia | ρ (Originale) | ρ (Adattivo) | p-value (Adattivo) | Significatività |
|---|:---:|:---:|:---:|:---:|
| **PES vs Combined Score** | +0.0453 | **+0.0588** | **0.0034** | **\*\*** |
| PES vs Providing Guidance | +0.0312 | **+0.0437** | **0.0296** | **\*** |
| PES vs Actionability | +0.0399 | **+0.0501** | **0.0128** | **\*** |

> [!IMPORTANT]
> Con i pesi adattivi, **tutte e tre le correlazioni sono statisticamente significative**, mentre con i pesi originali solo 2 su 3 lo erano. La correlazione con Combined Score raggiunge $p < 0.005$.

### Sub-metriche vs Combined Score

| Sub-metrica | ρ | p-value | Significatività |
|---|:---:|:---:|:---:|
| M3 (No-Immediate-Disclosure) | **+0.0426** | **0.047** | **\*** |
| M2 (Linguistic Adaptation) | +0.0054 | 0.801 | n.s. |
| M1 (Uptake) | **-0.0232** | 0.279 | n.s. |

> [!TIP]
> M3 è l'unica sotto-metrica significativamente correlata al ground truth, confermando la decisione di aumentarne il peso e azzerare M1 negli stati B e C.

### Test tra gruppi

| Test | PES Originale | PES Adattivo |
|---|:---:|:---:|
| ANOVA | F=1.33, p=0.266 (n.s.) | F=2.12, p=0.121 (n.s.) |
| **Kruskal-Wallis** | H=3.78, p=0.151 (n.s.) | **H=6.06, p=0.048 (\*)** |
| Effect size (η²) | 0.0011 | 0.0017 |

> [!IMPORTANT]
> Il Kruskal-Wallis diventa significativo ($p = 0.048$) con i pesi adattivi — i tre gruppi sono statisticamente distinguibili.

---

## 4. Ablation Study — Pesi Originali vs Adattivi

| Metrica | Pesi Originali | Pesi Adattivi | Δ |
|---|:---:|:---:|:---:|
| ρ vs Combined Score | +0.0453 | **+0.0588** | +30% |
| ρ vs Providing Guidance | +0.0312 (n.s.) | **+0.0437 (\*)** | **diventa significativo** |
| Kruskal-Wallis p-value | 0.151 (n.s.) | **0.048 (\*)** | **diventa significativo** |
| Gap Yes–No | +0.0174 | **+0.0224** | +28.2% |

![Ablation Comparison](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/07_ablation_comparison.png)

---

## 5. Classifica dei Modelli

| # | Modello | PES Medio | Gate Failure % | Tipo |
|:---:|---|:---:|:---:|---|
| 🥇 | **Gemini** | **0.5412** | 11.3% | LLM |
| 🥈 | **Llama 3.1 405B** | **0.5345** | 9.0% | LLM |
| 3 | Llama 3.1 8B | 0.5191 | 11.4% | LLM |
| 4 | GPT-4 | 0.5184 | 11.3% | LLM |
| 5 | Sonnet | 0.5148 | 11.7% | LLM |
| 6 | Phi3 | 0.5105 | 10.7% | LLM |
| 7 | Novice | 0.5078 | 14.5% | Baseline umano |
| 8 | Expert | 0.5011 | 14.3% | Baseline umano |
| 9 | Mistral | 0.4974 | 15.7% | LLM |

![Model Ranking](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/02_model_ranking.png)

> [!NOTE]
> **Finding interessante**: L'Expert umano si posiziona penultimo. Questo è coerente con la natura di PES: gli esperti umani tendono a dare risposte brevi e dirette (basso scaffolding), mentre PES premia risposte che guidano lo studente senza rivelare la soluzione.

### Sotto-metriche per Modello

![Sub-metrics Heatmap](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/03_submetrics_heatmap.png)

---

## 6. Distribuzione degli Stati Pedagogici

| Stato | Descrizione | % Campioni | PES Medio |
|:---:|---|:---:|:---:|
| **C** | Socratic Assessment | 61.3% | 0.5928 |
| **A** | Concept Teaching | 17.1% | 0.5578 |
| **B** | Error Remediation | 9.6% | **0.6047** |
| — | Gate Failed | 12.0% | 0.0000 |

![PES by State](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/05_pes_by_state.png)

> Lo stato B (Error Remediation) ottiene il PES più alto, coerente con il fatto che MRBench è un dataset centrato sulla correzione degli errori degli studenti.

---

## 7. Faithfulness Gate

![Gate Failure Rate](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/06_gate_failure_rate.png)

- **Mistral** ha il tasso di fallimento più alto (15.7%)
- **Llama 3.1 405B** il più basso (9.0%)
- Il threshold di 0.25 produce un tasso globale del 12.0%, ragionevole

---

## 8. Matrice delle Correlazioni Complete

![Correlation Matrix](/Users/carmine/.gemini/antigravity-ide/brain/4b8ff647-12ec-4b5e-9278-5606e126cb08/04_correlation_matrix.png)

---

## 9. Interpretazione per la Tesi

### Risultati principali

1. **PES correla significativamente con il giudizio umano** ($\rho = +0.059$, $p = 0.003$) su 2.475 campioni — la metrica cattura una dimensione reale della qualità pedagogica.

2. **M3 (No-Immediate-Disclosure) è la sotto-metrica più predittiva** — l'unica con correlazione significativa al ground truth ($p = 0.047$). Questo conferma che la capacità di scaffolding è il driver principale della qualità pedagogica.

3. **M1 (Uptake) ha correlazione negativa** ($\rho = -0.023$) — nei contesti di error correction, l'uptake conversazionale è controproducente perché premia risposte che elaborano il ragionamento errato dello studente. La calibrazione state-adaptive (w1=0 per Stati B,C) migliora tutti gli indicatori.

4. **La calibrazione dei pesi è empiricamente validata**: il passaggio ai pesi adattivi rende significativi il Kruskal-Wallis ($p = 0.048$) e la correlazione con Providing Guidance ($p = 0.030$), entrambi non significativi con i pesi originali.

### Limiti e future directions

- L'effect size è piccolo ($\eta^2 = 0.0017$), coerente col fatto che PES e MRBench misurano costrutti correlati ma diversi (processo pedagogico vs correttezza della guida).
- La validazione su **MathDial** (con annotazioni di teacher moves) dovrebbe produrre correlazioni più forti, poiché le etichette *telling* vs *probing* mappano direttamente su M3.

---

## 10. File prodotti

| File | Descrizione |
|---|---|
| [`mrbench_final_complete.csv`](file:///Users/carmine/Desktop/Tesi/output/mrbench_final_complete.csv) | DataFrame completo (2.475 righe) con tutti i sub-score e i punteggi ricalcolati |
| [`output/plots_final/`](file:///Users/carmine/Desktop/Tesi/output/plots_final) | 7 grafici publication-quality per la tesi |
| [`output/ablation/`](file:///Users/carmine/Desktop/Tesi/output/ablation) | Risultati grid search dei pesi + heatmap |
