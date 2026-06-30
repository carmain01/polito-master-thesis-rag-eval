"""LangChain integration for RAG evaluation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.documents import Document
except ImportError:
    raise ImportError(
        "Please install langchain-core to use the LangChain integration: "
        "pip install 'rag-eval[langchain]'"
    )

from rag_eval.core.types import TestSample

logger = logging.getLogger(__name__)


class RagEvalCallbackHandler(BaseCallbackHandler):
    """
    Callback handler to extract evaluation data from LangChain RAG pipelines.

    Attach this to your LangChain ``invoke()`` or ``ainvoke()`` calls to
    automatically capture queries, retrieved contexts (with full metadata),
    and the final generated answers — without modifying the chain itself.

    Usage::

        handler = RagEvalCallbackHandler()
        result = my_chain.invoke("question", config={"callbacks": [handler]})
        sample = handler.get_sample()   # -> TestSample ready for evaluation
    """

    def __init__(self) -> None:
        super().__init__()
        # Captured data keyed by root run_id
        self._runs: dict[UUID, dict[str, Any]] = {}
        # Maps every run_id (including children) to its root run_id
        self._run_map: dict[UUID, UUID] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _root_id(self, run_id: UUID, parent_run_id: UUID | None) -> UUID | None:
        """Resolve the root run_id for any given run."""
        if parent_run_id is None:
            return run_id
        return self._run_map.get(parent_run_id)

    def _store_documents(self, root_id: UUID, documents: Sequence[Document]) -> None:
        """Store documents with full metadata under the root run."""
        if root_id not in self._runs:
            return
        for doc in documents:
            self._runs[root_id]["retrieved_documents"].append({
                "page_content": doc.page_content,
                "metadata": dict(doc.metadata) if doc.metadata else {},
            })

    # ------------------------------------------------------------------
    # LangChain callback hooks
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Capture the user query when the root chain starts."""
        if parent_run_id is None:
            # Root chain — extract the question
            self._run_map[run_id] = run_id

            question = ""
            if isinstance(inputs, dict):
                for key in ("question", "query", "input"):
                    if key in inputs:
                        question = inputs[key]
                        break
                else:
                    question = str(inputs)
            elif isinstance(inputs, str):
                question = inputs

            self._runs[run_id] = {
                "question": question,
                "retrieved_documents": [],
                "answer": "",
                "chain_metadata": metadata or {},
            }
            logger.debug("Root chain started (run_id=%s), question=%r", run_id, question)
        else:
            # Child step — register it under the same root
            root_id = self._run_map.get(parent_run_id)
            if root_id:
                self._run_map[run_id] = root_id

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Capture retrieved documents (with metadata) from a Retriever step."""
        root_id = self._root_id(run_id, parent_run_id)
        if root_id and root_id in self._runs:
            self._store_documents(root_id, documents)
            logger.debug("Retriever captured %d documents (root=%s)", len(documents), root_id)

    def on_chain_end(
        self,
        outputs: dict[str, Any] | Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Capture contexts from intermediate steps and the final answer from the root chain."""
        root_id = self._root_id(run_id, parent_run_id)

        # 1. Intermediate step: if output contains Documents, capture them
        if root_id and root_id in self._runs and parent_run_id is not None:
            docs_to_check: list[Any] = []
            if isinstance(outputs, list):
                docs_to_check = outputs
            elif isinstance(outputs, dict):
                for val in outputs.values():
                    if isinstance(val, list):
                        docs_to_check = val
                        break

            captured = [item for item in docs_to_check if isinstance(item, Document)]
            if captured:
                self._store_documents(root_id, captured)
                logger.debug("Chain step captured %d documents (root=%s)", len(captured), root_id)

        # 2. Root chain end: capture the final answer
        if parent_run_id is None and run_id in self._runs:
            answer = ""
            if isinstance(outputs, dict):
                for key in ("answer", "result", "output"):
                    if key in outputs:
                        answer = outputs[key]
                        break
                else:
                    answer = str(outputs)
            elif isinstance(outputs, str):
                answer = outputs

            self._runs[run_id]["answer"] = answer
            logger.debug("Root chain ended (run_id=%s), answer=%r", run_id, answer[:80])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_samples(self) -> list[TestSample]:
        """
        Convert all captured runs into ``TestSample`` objects.

        The ``contexts`` field contains the plain-text page contents (what
        metrics like Faithfulness expect).  The full document metadata is
        preserved under ``sample.metadata["retrieved_documents"]`` for
        inspection and debugging.
        """
        samples: list[TestSample] = []
        for run_id, data in self._runs.items():
            # Deduplicate contexts (same doc can appear via retriever + chain_end)
            seen_contents: set[str] = set()
            unique_docs: list[dict[str, Any]] = []
            for doc in data["retrieved_documents"]:
                if doc["page_content"] not in seen_contents:
                    seen_contents.add(doc["page_content"])
                    unique_docs.append(doc)

            samples.append(TestSample(
                question=data["question"],
                answer=data["answer"],
                contexts=[d["page_content"] for d in unique_docs],
                metadata={
                    "langchain_run_id": str(run_id),
                    "retrieved_documents": unique_docs,
                    **data["chain_metadata"],
                },
            ))
        return samples

    def get_sample(self) -> TestSample | None:
        """Return the most recently captured ``TestSample``, or ``None``."""
        samples = self.get_samples()
        return samples[-1] if samples else None

    def reset(self) -> None:
        """Clear all captured runs."""
        self._runs.clear()
        self._run_map.clear()
