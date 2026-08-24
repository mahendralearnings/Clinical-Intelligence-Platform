"""
Evaluation runner - Phase 6. A standalone dev script, not part of the
API. Run manually to check if RAG answers are actually good, not just
that the code runs without crashing.

Makes REAL calls to your configured LLM provider - 8 questions x 2
calls each (1 real answer + 1 judge call) = 16 AI calls per run.
"""

import json
import sys
from pathlib import Path

# Add src to path so this script can import the app's code directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_platform.core.config import get_settings
from clinical_platform.infrastructure.embedding_providers.bedrock_embedding_provider import (
    BedrockEmbeddingProvider,
)
from clinical_platform.infrastructure.vector_stores.json_vector_store import JsonVectorStore
from clinical_platform.services.retrieval_service import RetrievalService
from clinical_platform.services.rag_query_service import RagQueryService

# Import whichever LLM provider is active, same way llm_dependencies.py does
from clinical_platform.infrastructure.llm_providers.bedrock_llm_provider import BedrockLLMProvider
from clinical_platform.infrastructure.llm_providers.anthropic_llm_provider import AnthropicLLMProvider
from clinical_platform.infrastructure.llm_providers.openai_llm_provider import OpenAILLMProvider

GOLDEN_DATASET_PATH = Path("data/golden_dataset.json")
RESULTS_PATH = Path("data/evaluation_results.jsonl")

I_DONT_KNOW = "I don't know based on the available documents."

JUDGE_PROMPT_TEMPLATE = """You are grading an AI assistant's answer for accuracy and groundedness.

Question: {question}

Retrieved source excerpts:
{sources}

AI's answer: {answer}

Rate this answer from 1-5 on accuracy and groundedness (does it stick to
the provided sources, or make things up?). Respond with ONLY valid JSON,
no other text: {{"score": <int 1-5>, "reason": "<one sentence>"}}
"""


def build_llm_provider(settings):
    provider_type = settings.llm_provider_type
    if provider_type == "anthropic":
        return AnthropicLLMProvider(api_key=settings.anthropic_api_key)
    if provider_type == "openai":
        return OpenAILLMProvider(api_key=settings.openai_api_key)
    return BedrockLLMProvider(region=settings.llm_region, model_id=settings.llm_model_id)


def build_rag_service(settings) -> RagQueryService:
    embedder = BedrockEmbeddingProvider(region=settings.bedrock_region)
    # store = JsonVectorStore(path=Path(settings.vector_store_path))
    
    store = JsonVectorStore(store_path=Path(settings.vector_store_path))
    retrieval = RetrievalService(embedder=embedder, store=store)
    llm = build_llm_provider(settings)
    # return RagQueryService(retrieval=retrieval, llm=llm, query_log_path=Path(settings.query_log_path))
    return RagQueryService(retrieval=retrieval, llm=llm, log_path=Path(settings.query_log_path))


def judge_answer(llm, question: str, answer: str, sources: list) -> dict:
    sources_text = "\n".join(
        f"- {s['source']} ({s.get('section_title', 'N/A')})" for s in sources
    ) or "(none retrieved)"
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, sources=sources_text, answer=answer)
    raw = llm.generate(prompt, max_tokens=150, temperature=0.0)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"score": 0, "reason": f"Could not parse judge response: {raw[:100]}"}


def run_evaluation() -> None:
    settings = get_settings()
    rag_service = build_rag_service(settings)
    judge_llm = build_llm_provider(settings)

    cases = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    results = []

    print(f"\nRunning evaluation on {len(cases)} questions using provider: {settings.llm_provider_type}\n")
    print(f"{'ID':<18} {'Retrieval':<10} {'Content':<10} {'Judge':<7} Reason")
    print("-" * 90)

    for case in cases:
        result = rag_service.query(question=case["question"], top_k=3)
        sources = [{"source": s.source, "section_title": s.section_title} for s in result.sources]

        if case["is_answerable"]:
            retrieval_pass = any(s["source"] == case["expected_source"] for s in sources)
            content_pass = any(
                kw.lower() in result.answer.lower() for kw in case["expected_keywords"]
            )
        else:
            retrieval_pass = len(sources) == 0
            content_pass = result.answer.strip() == I_DONT_KNOW

        judge = judge_answer(judge_llm, case["question"], result.answer, sources)

        print(
            f"{case['id']:<18} {'PASS' if retrieval_pass else 'FAIL':<10} "
            f"{'PASS' if content_pass else 'FAIL':<10} {judge['score']}/5    {judge['reason']}"
        )

        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answer": result.answer,
                "sources": sources,
                "retrieval_pass": retrieval_pass,
                "content_pass": content_pass,
                "judge_score": judge["score"],
                "judge_reason": judge["reason"],
            }
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(results)
    retrieval_passed = sum(1 for r in results if r["retrieval_pass"])
    content_passed = sum(1 for r in results if r["content_pass"])
    avg_judge = sum(r["judge_score"] for r in results) / total if total else 0

    print("-" * 90)
    print(f"Retrieval accuracy: {retrieval_passed}/{total}")
    print(f"Content accuracy:   {content_passed}/{total}")
    print(f"Average judge score: {avg_judge:.1f}/5")
    print(f"\nFull results saved to {RESULTS_PATH}\n")


if __name__ == "__main__":
    run_evaluation()