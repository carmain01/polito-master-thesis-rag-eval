# Synthetic Generation Methodology

This document outlines the methodology used for automatically generating synthetic question-answer (QA) pairs from reference documents. This synthetic generation pipeline is a central feature for evaluating Retrieval-Augmented Generation (RAG) frameworks in scenarios where labeled ground truth datasets are unavailable.

## 1. Document Chunking
The initial step in synthetic data generation involves splitting source documents into manageable chunks. The system implements a two-tier strategy:
- **Semantic-Aware Chunking (Primary):** The framework uses an embedding model (`sentence-transformers/paraphrase-MiniLM-L6-v2`) to segment text into semantically cohesive chunks. It analyzes sentence similarities and merges them until a similarity threshold is crossed or maximum chunk length is reached. This preserves context and prevents breaking ideas midway.
- **Fast Chunking (Fallback):** In case of embedding failure or extremely long documents, a fallback token-length or character-length based chunking is applied to guarantee processing continuity.

## 2. LLM-Based QA Generation
Once chunks are created, an LLM is prompted to act as an expert dataset creator. The generator supports various question types, selectable dynamically based on evaluation goals:
- **Factual:** Straightforward extraction of facts present in the text.
- **Multi-hop:** Questions that require combining information spread across different sentences in the chunk.
- **Reasoning:** Questions that ask for logically deducible inferences not explicitly stated.
- **Comparative:** Questions comparing entities or concepts found within the chunk.

## 3. Difficulty Modulation
The prompt generation engine modifies the underlying instruction with a "difficulty modifier" (e.g., Easy, Medium, Hard). This dynamically steers the LLM to either generate explicit queries (Easy) or queries that require deeper synthesis and paraphrasing (Hard).

## 4. Context Handling and Distractor Injection
Each generated QA pair is associated with the source chunk as its "ground truth context". To evaluate the resilience of retriever modules in RAG systems, the pipeline implements **Distractor Context Injection**. This injects plausible but irrelevant chunks from the same document alongside the actual source chunk. The retriever must demonstrate the ability to distinguish the true context from the distractor.

## 5. Quality Assurance and Deduplication
To ensure dataset rigor, generated pairs undergo automated quality checks before finalization:
- **Quality Filtering:** Any generated pairs with empty questions/answers, excessively short text (e.g., questions under 10 characters), or questions that are identical to their answers are discarded.
- **Deduplication:** The pipeline maintains a runtime state of normalized generated questions. If the LLM generates a semantically identical or exact duplicate question across different chunks, the duplicate is ignored to ensure dataset diversity.

## Conclusion
This controlled synthetic generation pipeline provides a cost-effective, scalable, and rigorous method for generating diverse evaluation datasets. It forms the backbone of the experimental evaluation strategy detailed in the thesis.
