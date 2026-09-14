# RAG Evaluation Framework — Documentation Index

Benvenuto nella documentazione ufficiale del framework **`rag_eval`**, un toolkit modulare, asincrono ed estendibile per la valutazione qualitativa, quantitativa e pedagogica di sistemi **Retrieval-Augmented Generation (RAG)** in ambienti accademici e di e-learning.

---

## 📚 Mappa dei Documenti

La cartella `docs/` include la documentazione tecnica e metodologica di riferimento del framework:

| Documento | Descrizione sintetica | Destinatari |
| :--- | :--- | :--- |
| [**`SYSTEM_OVERVIEW.md`**](file:///Users/carmine/Desktop/Tesi/docs/SYSTEM_OVERVIEW.md) | **Architettura complessiva del sistema**: design dei componenti, modelli dati Pydantic, sistema di metriche (RAG Triad, NLP, Embedding e la nuova metrica composita **PES - Pedagogical Evaluation Score**), provider LLM supportati (incluso Azure OpenAI), gestione della cache e resilienza a checkpoint. | Ricercatori, Sviluppatori, Revisori |
| [**`cli.md`**](file:///Users/carmine/Desktop/Tesi/docs/cli.md) | **Manuale di riferimento della CLI (`rag-eval`)**: guida esaustiva a tutti i comandi da terminale (`evaluate`, `generate`, `report`, `metrics`), opzioni di override (`--provider`, `--model`, `--reasoning-effort`), formati di export (Console, JSON, HTML Dashboard con Chart.js, CSV) ed esempi pratici. | Utenti CLI, Sperimentatori |
| [**`SYNTHETIC_GENERATION.md`**](file:///Users/carmine/Desktop/Tesi/docs/SYNTHETIC_GENERATION.md) | **Metodologia di Generazione Sintetica**: pipeline per la creazione automatica di dataset QA da testi grezzi, chunking semantico via embedding, iniezione di contesti distrattori e simulazione stocastica dei comportamenti RAG (risposte fedeli, allucinazioni, astensione). | Ricercatori, Data Creator |

---

## 🧭 Panoramica Rapida del Framework

### 1. Sistema di Metriche
Il framework integra tre livelli complementari di analisi:
1. **RAG Triad Classica**: `Faithfulness`, `Answer Relevance`, `Context Precision`, `Context Recall` (valutate tramite LLM-as-a-Judge con prompt strutturati).
2. **Pedagogical Evaluation Score (PES)**: metrica composita ideata per assistenti didattici e dialoghi tutoriali, basata su media geometrica ponderata condizionata allo stato pedagogico (*Concept Teaching*, *Error Remediation*, *Socratic Assessment*) e composta dalle sotto-metriche:
   - **$M_1$ Uptake**: valorizzazione e continuità degli input dello studente (NUC-BERT).
   - **$M_2$ Linguistic Adaptation**: calibrazione del registro, tentatività socratica e tono costruttivo.
   - **$M_3$ No-Immediate-Disclosure**: qualità dello *scaffolding* didattico ed astensione dallo *spoiling*.
   - **Gate di Faithfulness**: annullamento a $0.0$ del punteggio didattico se la risposta contiene allucinazioni.
3. **Metriche NLP ed Embedding**: `BLEU`, `ROUGE` (1/2/L), `Token F1`, `Exact Match`, `Semantic Similarity`, `BERTScore`.

### 2. Provider LLM Supportati
- **Cloud Commerciali**: Azure OpenAI (inclusi modelli reasoning `gpt-5-nano`, `o3-mini`), OpenAI (`gpt-4o`, `gpt-4o-mini`), Anthropic (`claude-3-5-sonnet`, `haiku`), Google (`gemini-2.5-pro`, `flash`).
- **Locali / On-Premise**: Ollama (`llama3.1`, `llama3.2`, `qwen2.5`) e vLLM (API OpenAI-compatibile ad alto throughput).

### 3. Ingestion Dati & Benchmark
Supporto integrato per dataset JSON, JSONL, CSV, benchmark standard QA (*HotpotQA, Natural Questions, TriviaQA, SQuAD 2.0, RGB, RECALL*) e dataset didattici specializzati (*MathDial, MRBench v3*).
