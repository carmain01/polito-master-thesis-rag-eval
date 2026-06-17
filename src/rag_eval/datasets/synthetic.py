"""Synthetic Data Generator for RAG evaluation."""

from __future__ import annotations

import logging
import random

from rag_eval.core.config import LLMConfig
from rag_eval.core.types import TestSample
from rag_eval.utils.chunking import ChunkingService
from rag_eval.utils.llm import LLMClient

logger = logging.getLogger(__name__)

PROMPT_TEMPLATES = {
    "factual": "Generate a straightforward factual question based on the following text. Provide the question and the exact answer.",
    "multi-hop": "Generate a multi-hop question that requires combining information from different parts of the text. Provide the question and the answer.",
    "reasoning": "Generate a reasoning question that requires inferring information not explicitly stated, but logically deducible from the text. Provide the question and the answer.",
    "comparative": "Generate a question that asks to compare two or more entities or concepts mentioned in the text. Provide the question and the answer."
}

class SyntheticDataGenerator:
    """Generates synthetic QA pairs from documents for RAG evaluation."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        if llm_client is None:
            # Default to local model (Ollama) as requested
            config = LLMConfig(provider="ollama", model="llama3")
            self.llm_client = LLMClient(config=config)
        else:
            self.llm_client = llm_client

        self.chunking_service = ChunkingService()



    async def generate_qa_pairs(
        self,
        text: str,
        num_questions_per_chunk: int = 1,
        question_types: list[str] | None = None,
        difficulty: str = "medium"
    ) -> list[TestSample]:
        """Generate synthetic QA pairs from a given document text."""
        chunks = self.chunking_service.semantic_aware_chunking(text)
        if not chunks:
            return []

        if not question_types:
            question_types = ["factual"]

        samples = []
        seen_questions = set()

        system_prompt = (
            "You are an expert dataset creator. Output valid JSON in the format: "
            "{\"question\": \"...\", \"answer\": \"...\"}"
        )

        for chunk_id, chunk in enumerate(chunks):
            for _ in range(num_questions_per_chunk):
                q_type = random.choice(question_types)
                instruction = PROMPT_TEMPLATES.get(q_type, PROMPT_TEMPLATES["factual"])

                # Add difficulty modifier to instruction
                difficulty_mod = f" Make the question of {difficulty} difficulty."

                prompt = f"{instruction}{difficulty_mod}\n\nText Context:\n{chunk}"

                try:
                    result_json = await self.llm_client.complete_json(
                        prompt=prompt,
                        system=system_prompt
                    )

                    if "question" in result_json and "answer" in result_json:
                        q_text = result_json["question"].strip()
                        a_text = result_json["answer"].strip()

                        # Quality Filtering
                        if not q_text or not a_text:
                            logger.debug(f"Filtered chunk {chunk_id}: empty question or answer.")
                            continue
                        if len(q_text) < 10 or len(a_text) < 2:
                            logger.debug(f"Filtered chunk {chunk_id}: question or answer too short.")
                            continue
                        if q_text.lower() == a_text.lower():
                            logger.debug(f"Filtered chunk {chunk_id}: question identical to answer.")
                            continue

                        # Deduplication
                        q_norm = q_text.lower()
                        if q_norm in seen_questions:
                            logger.debug(f"Filtered chunk {chunk_id}: duplicate question.")
                            continue
                        seen_questions.add(q_norm)

                        # Distractor context injection (dummy for now, just random other chunk if available)
                        distractor = ""
                        if len(chunks) > 1:
                            other_chunks = [c for i, c in enumerate(chunks) if i != chunk_id]
                            distractor = random.choice(other_chunks)

                        contexts = [chunk]
                        if distractor:
                            contexts.append(distractor)
                            random.shuffle(contexts)

                        sample = TestSample(
                            question=q_text,
                            ground_truth=a_text,
                            contexts=contexts,
                            metadata={
                                "source": "synthetic",
                                "question_type": q_type,
                                "difficulty": difficulty,
                                "chunk_id": chunk_id,
                                "is_synthetic": True
                            }
                        )
                        samples.append(sample)
                except Exception as e:
                    logger.error(f"Failed to generate QA pair for chunk {chunk_id}: {e}")

        return samples
