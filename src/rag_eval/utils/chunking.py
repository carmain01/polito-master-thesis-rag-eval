import re

import numpy as np

_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "inc",
    "ltd",
    "corp",
    "dept",
    "univ",
    "govt",
    "approx",
    "assn",
    "bros",
    "co",
    "no",
    "vol",
    "rev",
    "gen",
    "gov",
    "sgt",
    "cpl",
    "pvt",
    "capt",
    "lt",
    "cmdr",
    "adm",
    "maj",
    "col",
    "brig",
    "fig",
    "figs",
    "eq",
    "eqs",
    "al",
    "e",
    "i",
}

# Aggressive split: sentence-ending punctuation followed by whitespace + uppercase
_SENT_SPLIT = re.compile(r'([.!?])(\s+)(?=[A-Z\d"\'\(\[])')

def get_chunking_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")


def _tokenize_paragraph(para: str) -> list[str]:
    tokens = _SENT_SPLIT.split(para)
    raw_sents: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens):
            raw_sents.append(tokens[i] + tokens[i + 1])
            i += 3
        else:
            raw_sents.append(tokens[i])
            i += 1
    return raw_sents

def _rejoin_abbreviations_and_decimals(raw_sents: list[str]) -> list[str]:
    merged: list[str] = []
    for sent in raw_sents:
        sent = sent.strip()
        if not sent:
            continue
        if merged:
            prev = merged[-1]
            last_word = re.split(r"\s+", prev)[-1] if prev else ""
            bare = last_word.rstrip(".!?").lower()
            if bare in _ABBREVIATIONS or re.search(r"\d\.$", prev):
                merged[-1] = prev + " " + sent
                continue
        merged.append(sent)
    return merged


def _split_sentences(text: str) -> list[str]:

    paragraphs = re.split(r"\n\s*\n", text)
    sentences: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        raw_sents = _tokenize_paragraph(para)
        merged = _rejoin_abbreviations_and_decimals(raw_sents)
        sentences.extend(merged)

    # Fallback: if regex produced nothing useful, split on single newlines
    if not sentences:
        sentences = [s.strip() for s in text.split("\n") if s.strip()]

    # Last resort: return the whole text as one sentence
    if not sentences:
        sentences = [text.strip()] if text.strip() else []

    return sentences


class ChunkingService:
    def __init__(self):
        self.semantic_model = get_chunking_model()

    def chunk_text(self, text: str, max_chunk_chars: int = 1500) -> list[str]:
        """Keep parser records intact unless they are too long, then split semantically."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= max_chunk_chars:
            return [text]

        try:
            chunks = self.semantic_aware_chunking(
                text,
                similarity_threshold=0.3,
                min_chunk_chars=100,
                max_chunk_chars=max_chunk_chars,
                overlap=1,
            )
            if chunks:
                return chunks
        except Exception as e:
            print(f"[ChunkingService] Semantic chunking failed, using fallback: {e}")

        return self.fast_chunking(text)

    def fast_chunking(self, text: str) -> list[str]:
        """
        Fallback rapido basato su lunghezza del testo.
        """
        return [text[i : i + 800] for i in range(0, len(text), 800)]

    def _clean_sentences(self, sentences: list[str]) -> list[str]:
        cleaned = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) > 10 or (s and s[0].isupper()):
                cleaned.append(s)
        return cleaned

    def _compute_centroid_similarity(self, embeddings: np.ndarray, current_indices: list[int], target_embedding: np.ndarray) -> float:
        centroid = np.mean(embeddings[list(current_indices)], axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        return float(np.dot(centroid_norm, target_embedding))

    def _should_split(
        self,
        sim: float,
        current_length: int,
        candidate_length: int,
        similarity_threshold: float,
        min_chunk_chars: int,
        max_chunk_chars: int,
    ) -> bool:
        if sim < similarity_threshold and current_length >= min_chunk_chars:
            return True
        if candidate_length > max_chunk_chars and current_length >= min_chunk_chars:
            return True
        return False

    def semantic_aware_chunking(
        self,
        text: str,
        similarity_threshold: float = 0.5,
        min_chunk_chars: int = 100,
        max_chunk_chars: int = 1500,
        overlap: int = 1,
    ) -> list[str]:

        sentences = _split_sentences(text)
        cleaned = self._clean_sentences(sentences)

        if not cleaned:
            return [text] if text.strip() else []

        if len(cleaned) == 1:
            return [cleaned[0]]

        embeddings = self.semantic_model.encode(cleaned, normalize_embeddings=True)

        chunks: list[str] = []
        current_indices: list[int] = [0]

        for i in range(1, len(cleaned)):
            sim = self._compute_centroid_similarity(embeddings, current_indices, embeddings[i])

            current_text = " ".join(cleaned[j] for j in current_indices)
            candidate_text = current_text + " " + cleaned[i]

            should_split = self._should_split(
                sim,
                len(current_text),
                len(candidate_text),
                similarity_threshold,
                min_chunk_chars,
                max_chunk_chars,
            )

            if should_split:
                chunks.append(current_text)
                # Overlap: riprendi le ultime `overlap` frasi nel nuovo chunk
                if overlap > 0 and len(current_indices) >= overlap:
                    current_indices = list(current_indices[-overlap:]) + [i]
                else:
                    current_indices = [i]
            else:
                current_indices.append(i)

        # Ultimo chunk
        last_text = " ".join(cleaned[j] for j in current_indices)
        if chunks and len(last_text) < min_chunk_chars:
            # Se l'ultimo chunk è troppo piccolo, uniscilo al precedente
            chunks[-1] = chunks[-1] + " " + last_text
        else:
            chunks.append(last_text)

        return chunks
