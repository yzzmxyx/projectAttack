"""Reference trajectory matching helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .types import ReferencePrototype


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, band: Optional[int] = None) -> float:
    a = np.asarray(seq_a, dtype=np.float32)
    b = np.asarray(seq_b, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")

    band_width = max(a.shape[0], b.shape[0]) if band is None else max(1, int(band))
    dp = np.full((a.shape[0] + 1, b.shape[0] + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, a.shape[0] + 1):
        j_start = max(1, i - band_width)
        j_end = min(b.shape[0], i + band_width)
        for j in range(j_start, j_end + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[a.shape[0], b.shape[0]])


def match_reference_prototypes(
    feature_history: Iterable[np.ndarray],
    prototypes: Iterable[ReferencePrototype],
    topk: int = 3,
    band: int = 4,
    cosine_weight: float = 0.7,
) -> Optional[dict]:
    history = [np.asarray(item, dtype=np.float32).reshape(-1) for item in feature_history]
    if not history:
        return None

    query = np.stack(history, axis=0)
    query_last = query[-1]
    candidates = []

    for prototype in prototypes:
        proto_features = np.asarray(prototype.feature_vectors, dtype=np.float32)
        if proto_features.ndim != 2 or proto_features.shape[0] == 0:
            continue
        progress_points = list(prototype.progress_points)
        if len(progress_points) == 0:
            progress_points = [float(idx) / float(max(1, proto_features.shape[0] - 1)) for idx in range(proto_features.shape[0])]

        cosine_scores = np.asarray([cosine_similarity(query_last, row) for row in proto_features], dtype=np.float32)
        best_idx = int(np.argmax(cosine_scores))
        best_cosine = float(cosine_scores[best_idx])

        if query.shape[0] > 1 and proto_features.shape[0] > 1:
            tail_len = min(query.shape[0], best_idx + 1)
            query_tail = query[-tail_len:]
            proto_tail = proto_features[best_idx - tail_len + 1 : best_idx + 1]
            dtw = dtw_distance(query_tail, proto_tail, band=band)
            dtw_score = 1.0 / (1.0 + float(dtw))
        else:
            dtw_score = best_cosine

        score = (float(cosine_weight) * best_cosine) + ((1.0 - float(cosine_weight)) * dtw_score)
        candidates.append(
            {
                "prototype": prototype,
                "score": float(score),
                "anchor_progress": float(progress_points[min(best_idx, len(progress_points) - 1)]),
                "best_idx": best_idx,
                "cosine": best_cosine,
                "dtw_score": float(dtw_score),
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[: max(1, int(topk))]
    total_weight = sum(max(1e-6, float(item["score"])) for item in selected)
    anchor_progress = sum(float(item["anchor_progress"]) * max(1e-6, float(item["score"])) for item in selected) / total_weight
    best = selected[0]
    return {
        "anchor_progress": float(np.clip(anchor_progress, 0.0, 1.0)),
        "score": float(best["score"]),
        "phase_id": str(best["prototype"].phase_id),
        "prototype_task_id": str(best["prototype"].task_id),
        "cosine": float(best["cosine"]),
        "dtw_score": float(best["dtw_score"]),
    }
