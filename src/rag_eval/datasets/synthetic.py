"""Synthetic Data Generator for RAG evaluation."""

from __future__ import annotations

import json
import logging
import random
from typing import Any, List

from rag_eval.core.config import LLMConfig
from rag_eval.core.types import TestSample
from rag_eval.utils.llm import LLMClient
from rag_eval.utils.chunking import ChunkingService

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
        question_types: List[str] | None = None,
        difficulty: str = "medium"
    ) -> List[TestSample]:
        """Generate synthetic QA pairs from a given document text."""
        chunks = self.chunking_service.semantic_aware_chunking(text)
        if not chunks:
            return []

        if not question_types:
            question_types = ["factual"]

        samples = []
        
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
                            question=result_json["question"],
                            ground_truth=result_json["answer"],
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
