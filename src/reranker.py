"""Обучение и применение линейного reranker для кандидатов поиска."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.retrieval import SOURCE_DEPTH, Source, row_candidates

NEGATIVES_PER_POSITIVE = 25
SEED = 42


def candidate_matrix(
    source_rows: list[tuple[np.ndarray, np.ndarray]],
    candidates: np.ndarray,
    search_location: object,
    item_locations: np.ndarray,
    query_length: int,
    title_lengths: np.ndarray,
) -> np.ndarray:
    """Строит матрицу признаков для кандидатов поискового контекста."""
    scores: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    flags: list[np.ndarray] = []

    for ids, values in source_rows:
        valid = ids >= 0
        positions = np.searchsorted(
            candidates,
            ids[valid],
        )

        score = np.zeros(
            len(candidates),
            dtype=np.float32,
        )
        rank = np.full(
            len(candidates),
            SOURCE_DEPTH + 1,
            dtype=np.float32,
        )
        flag = np.zeros(
            len(candidates),
            dtype=np.float32,
        )

        score[positions] = values[valid]
        rank[positions] = np.arange(
            1,
            valid.sum() + 1,
        )
        flag[positions] = 1

        scores.append(score)
        ranks.append(rank)
        flags.append(flag)

    source_count = np.sum(
        flags,
        axis=0,
    )

    location_match = (item_locations[candidates] == search_location).astype(np.float32)

    query_lengths = np.full(
        len(candidates),
        query_length,
        dtype=np.float32,
    )

    candidate_title_lengths = title_lengths[candidates].astype(np.float32)

    return np.column_stack(
        [
            *scores,
            *ranks,
            *flags,
            source_count,
            location_match,
            query_lengths,
            candidate_title_lengths,
        ]
    )


def build_training_data(
    contexts: pd.DataFrame,
    truth: dict[str, frozenset[str]],
    sources: dict[str, Source],
    item_index: dict[str, int],
    item_locations: np.ndarray,
    title_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Формирует обучающую выборку с негативным семплированием."""
    generator = np.random.default_rng(SEED)

    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for row, key in enumerate(contexts.index):
        candidates, source_rows = row_candidates(
            sources,
            row,
        )

        positives = np.asarray(
            [item_index[item] for item in truth[key]],
            dtype=np.int32,
        )

        found = np.intersect1d(
            candidates,
            positives,
            assume_unique=True,
        )

        if not len(found):
            continue

        negative_pool = np.setdiff1d(
            candidates,
            found,
            assume_unique=True,
        )

        negative_count = min(
            len(negative_pool),
            NEGATIVES_PER_POSITIVE * len(found),
        )

        if negative_count:
            negatives = generator.choice(
                negative_pool,
                negative_count,
                replace=False,
            )
        else:
            negatives = np.empty(
                0,
                dtype=np.int32,
            )

        context = contexts.iloc[row]

        matrix = candidate_matrix(
            source_rows=source_rows,
            candidates=candidates,
            search_location=context.search_location_id,
            item_locations=item_locations,
            query_length=len(str(context.search_query)),
            title_lengths=title_lengths,
        )

        selected = np.concatenate(
            [
                found,
                negatives,
            ]
        )

        selected_positions = np.searchsorted(
            candidates,
            selected,
        )

        blocks.append(matrix[selected_positions])
        labels.append(
            np.concatenate(
                [
                    np.ones(
                        len(found),
                        dtype=np.int8,
                    ),
                    np.zeros(
                        len(negatives),
                        dtype=np.int8,
                    ),
                ]
            )
        )

    if not blocks:
        raise ValueError('No positive items were found in the candidate pool')

    return (
        np.concatenate(blocks),
        np.concatenate(labels),
    )


def fit_reranker(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[StandardScaler, LogisticRegression]:
    """Обучает LogisticRegression с балансировкой классов."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = LogisticRegression(
        class_weight='balanced',
        C=1.0,
        solver='lbfgs',
        max_iter=200,
        random_state=SEED,
    )
    model.fit(
        scaled_features,
        labels,
    )

    return scaler, model


def predict_top50(
    contexts: pd.DataFrame,
    sources: dict[str, Source],
    item_locations: np.ndarray,
    title_lengths: np.ndarray,
    item_ids: np.ndarray,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> list[list[str]]:
    """Ранжирует объединённый пул кандидатов и возвращает top-50."""
    result: list[list[str]] = []

    for row in range(len(contexts)):
        candidates, source_rows = row_candidates(
            sources,
            row,
        )

        context = contexts.iloc[row]

        matrix = candidate_matrix(
            source_rows=source_rows,
            candidates=candidates,
            search_location=context.search_location_id,
            item_locations=item_locations,
            query_length=len(str(context.search_query)),
            title_lengths=title_lengths,
        )

        scaled_features = scaler.transform(matrix)
        probabilities = model.predict_proba(
            scaled_features,
        )[:, 1]

        order = np.lexsort(
            (
                candidates,
                -probabilities,
            )
        )

        top_candidates = candidates[order[:50]]

        result.append(item_ids[top_candidates].tolist())

    return result
