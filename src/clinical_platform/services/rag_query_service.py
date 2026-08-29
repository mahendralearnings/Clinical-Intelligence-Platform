"""RAG Query Service — Phase 5.

Wires RetrievalService (Phase 3) and LLMProvider (Phase 4):
  1. Search for relevant chunks above a similarity threshold.
  2. If none found, return "I don't know" immediately — no LLM call made.
  3. Otherwise build a grounded prompt and call Claude.
  4. Append one line to a JSONL log file.
  5. Return RagResult(answer, sources).

Zero direct imports from boto3, json store, or any infrastructure package.
All I/O is via injected collaborators or stdlib (datetime, json, pathlib).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from clinical_platform.domain.chunk import DocumentChunk
from clinical_platform.domain.ports import LLMProvider
from clinical_platform.domain.rag import RagResult, SourceCitation
from clinical_platform.services.retrieval_service import RetrievalService

from clinical_platform.services.guardrail_service import check_for_prompt_injection, redact_pii



#for langsmith
from langsmith import traceable


# Exact string — detectable programmatically in tests and eval pipelines
_NO_KNOWLEDGE_ANSWER = "I don't know based on the available documents."

_SYSTEM_PROMPT = (
    "You are a clinical intelligence assistant. "
    "Answer questions strictly using the document excerpts provided below. "
    "If the answer is not contained in the excerpts, respond with exactly: "
    "'I don't know based on the available documents.' "
    "Do not use any general knowledge or information outside the excerpts."
)

# Chunks below this cosine similarity are considered irrelevant
_MIN_SCORE = 0.5


class RagQueryService:
    """Combines retrieval and LLM generation into a single query operation.

    Args:
        retrieval: RetrievalService from Phase 3.
        llm:       LLMProvider from Phase 4.
        log_path:  Path to the JSONL query log file. Parent directories
                   are created automatically on first write.
    """

    def __init__(
        self,
        retrieval: RetrievalService,
        llm: LLMProvider,
        log_path: Path,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
   
    @traceable(name="rag_query")   ##adding for langsmith tracablity

    def query(self, question: str, top_k: int = 3) -> RagResult:
        """Answer *question* using retrieved document chunks.

        Steps:
            1. Retrieve top-K chunks with min_score=0.5.
            2. If no chunks pass the threshold, return "I don't know"
               immediately — no LLM call is made (saves cost + latency).
            3. Build a grounded prompt and call Claude with temperature=0.0.
            4. Append one line to the JSONL log file.
            5. Return RagResult(answer, sources).
        """
        
        # Step 0 — guardrail checks, before anything else
        check_for_prompt_injection(question)
        question, _pii_found = redact_pii(question)

     
        # Step 1 — retrieve relevant chunks
        results = self._retrieval.search(
            query=question,
            top_k=top_k,
            min_score=_MIN_SCORE,
        )

        # Step 2 — short-circuit if nothing relevant found
        if not results:
            self._write_log(question, [], _NO_KNOWLEDGE_ANSWER)
            return RagResult(answer=_NO_KNOWLEDGE_ANSWER, sources=[])

        # Step 3 — build prompt and call LLM
        prompt = self._build_prompt(question, results)
        answer = self._llm.generate(
            prompt,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=512,
            temperature=0.0,
        )

        # Step 4 — build citations
        sources = [
            SourceCitation(
                source=chunk.metadata.source,
                section_title=chunk.metadata.section_title,
                score=round(score, 6),
            )
            for chunk, score in results
        ]

        # Step 5 — log and return
        self._write_log(question, results, answer)
        return RagResult(answer=answer, sources=sources)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        question: str,
        results: list[tuple[DocumentChunk, float]],
    ) -> str:
        """Format retrieved chunks into the numbered excerpt prompt."""
        excerpts = []
        for i, (chunk, _score) in enumerate(results, start=1):
            section = chunk.metadata.section_title or "N/A"
            excerpts.append(
                f"[{i}] Source: {chunk.metadata.source} | Section: {section}\n"
                f"{chunk.content}"
            )
        excerpt_block = "\n\n".join(excerpts)
        return f"Question: {question}\n\nDocument excerpts:\n{excerpt_block}"

    def _write_log(
        self,
        question: str,
        results: list[tuple[DocumentChunk, float]],
        answer: str,
    ) -> None:
        """Append one JSON line to the query log file.

        Creates parent directories and the file if absent.
        Append mode — each write is one complete JSON object + newline.
        """
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "question": question,
            "retrieved_chunks": [
                {
                    "source": chunk.metadata.source,
                    "section_title": chunk.metadata.section_title,
                    "score": round(score, 6),
                }
                for chunk, score in results
            ],
            "answer": answer,
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
