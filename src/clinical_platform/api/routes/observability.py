"""
Observability endpoint - reads the existing query log file and reports
simple counts. Deliberately a plain function, not a service class -
this is just reading and counting a file, doesn't need the full
layered architecture.
"""

import json
from collections import Counter
from pathlib import Path
from fastapi import APIRouter

from clinical_platform.api.schemas.observability import ObservabilitySummary
from typing import Annotated

from fastapi import Depends

from clinical_platform.api.middleware.auth_dependencies import CurrentUser
from clinical_platform.core.config import Settings, get_settings



router = APIRouter(prefix="/observability", tags=["observability"])

I_DONT_KNOW_ANSWER = "I don't know based on the available documents."


@router.get("/summary", response_model=ObservabilitySummary)
def get_summary(
    settings: Annotated[Settings, Depends(get_settings)],
    _: CurrentUser,
) -> ObservabilitySummary:
    log_path = Path(settings.query_log_path)

    if not log_path.exists():
        return ObservabilitySummary(
            total_queries=0,
            answered_from_documents=0,
            said_i_dont_know=0,
            average_top_score=0.0,
            most_common_source_document=None,
        )

    total = 0
    dont_know_count = 0
    top_scores: list[float] = []
    source_counter: Counter[str] = Counter()

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            total += 1

            if entry.get("answer", "").strip() == I_DONT_KNOW_ANSWER:
                dont_know_count += 1

            chunks = entry.get("retrieved_chunks", [])
            if chunks:
                scores = [c.get("score", 0.0) for c in chunks]
                top_scores.append(max(scores))
                for c in chunks:
                    source = c.get("source")
                    if source:
                        source_counter[source] += 1

    average_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    most_common = source_counter.most_common(1)
    most_common_source = most_common[0][0] if most_common else None

    return ObservabilitySummary(
        total_queries=total,
        answered_from_documents=total - dont_know_count,
        said_i_dont_know=dont_know_count,
        average_top_score=round(average_top_score, 3),
        most_common_source_document=most_common_source,
    )