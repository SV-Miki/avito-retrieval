"""Формирование кандидатов из лексических и семантических источников."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.lexical import LexicalConfig, LexicalIndex, top_k_indices
from src.semantic import encode_unique_queries, top_k_dot_product

SOURCE_DEPTH = 100
SOURCE_NAMES = ('char', 'bm25', 'e5', 'params', 'description')


@dataclass(frozen=True)
class Source:
    """Результаты одного источника кандидатов для набора контекстов."""

    name: str
    indices: np.ndarray
    scores: np.ndarray


def field_text(
    frame: pd.DataFrame,
    column: str,
) -> list[str]:
    """Возвращает текстовый столбец без пропусков, сохраняя порядок строк."""
    return frame[column].fillna('').astype(str).tolist()


def params_queries(contexts: pd.DataFrame) -> list[str]:
    """Формирует запросы для поиска по параметрам объявлений."""
    queries = field_text(contexts, 'search_query')
    params = field_text(contexts, 'search_infm_params_text')

    return [
        query if not value.strip() else f'{query} {value}'
        for query, value in zip(queries, params, strict=False)
    ]


def lexical_topk_with_scores(
    index: LexicalIndex,
    texts: list[str],
    k: int = SOURCE_DEPTH,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает top-K лексических результатов вместе с оценками."""
    unique_texts, inverse = np.unique(
        texts,
        return_inverse=True,
    )

    result_ids = np.full(
        (len(unique_texts), k),
        -1,
        dtype=np.int32,
    )
    result_scores = np.zeros(
        (len(unique_texts), k),
        dtype=np.float32,
    )

    for begin in range(0, len(unique_texts), batch_size):
        end = min(begin + batch_size, len(unique_texts))
        queries = index.vectorizer.transform(
            unique_texts[begin:end],
        )

        if index.config.model == 'bm25':
            queries.data.fill(1)

        scores = (queries @ index.postings).tocsr()

        for row in range(end - begin):
            left, right = scores.indptr[row : row + 2]
            columns = scores.indices[left:right]
            values = scores.data[left:right]

            selected = top_k_indices(
                columns,
                values,
                k,
            )

            if len(selected):
                order = np.argsort(
                    columns,
                    kind='stable',
                )
                positions = np.searchsorted(
                    columns[order],
                    selected,
                )

                result_ids[
                    begin + row,
                    : len(selected),
                ] = selected

                result_scores[
                    begin + row,
                    : len(selected),
                ] = values[order[positions]]

    return result_ids[inverse], result_scores[inverse]


def e5_topk_with_scores(
    texts: list[str],
    item_embeddings: np.ndarray,
    encoder: SentenceTransformer,
) -> tuple[np.ndarray, np.ndarray]:
    """Кодирует запросы и выполняет семантический top-K поиск."""
    embeddings, inverse = encode_unique_queries(
        encoder,
        texts,
    )

    indices, scores = top_k_dot_product(
        embeddings,
        item_embeddings,
        SOURCE_DEPTH,
    )

    return indices[inverse], scores[inverse]


def retrieve_sources(
    contexts: pd.DataFrame,
    item_ids: np.ndarray,
    titles: list[str],
    params: list[str],
    descriptions: list[str],
    encoder: SentenceTransformer,
    item_embeddings: np.ndarray,
) -> dict[str, Source]:
    """Строит пять источников кандидатов для поисковых контекстов."""
    queries = field_text(
        contexts,
        'search_query',
    )

    configurations = (
        (
            'char',
            LexicalIndex(
                LexicalConfig(
                    model='tfidf',
                    analyzer='char_wb',
                    ngram_range=(3, 5),
                    min_df=2,
                )
            ).fit(
                item_ids,
                titles,
            ),
            queries,
        ),
        (
            'bm25',
            LexicalIndex(
                LexicalConfig(
                    model='bm25',
                )
            ).fit(
                item_ids,
                titles,
            ),
            queries,
        ),
        (
            'params',
            LexicalIndex(
                LexicalConfig(
                    model='bm25',
                )
            ).fit(
                item_ids,
                params,
            ),
            params_queries(contexts),
        ),
        (
            'description',
            LexicalIndex(
                LexicalConfig(
                    model='tfidf',
                )
            ).fit(
                item_ids,
                descriptions,
            ),
            queries,
        ),
    )

    sources: dict[str, Source] = {}

    for name, index, texts in configurations:
        indices, scores = lexical_topk_with_scores(
            index,
            texts,
        )
        sources[name] = Source(
            name=name,
            indices=indices,
            scores=scores,
        )

    indices, scores = e5_topk_with_scores(
        queries,
        item_embeddings,
        encoder,
    )
    sources['e5'] = Source(
        name='e5',
        indices=indices,
        scores=scores,
    )

    return sources


def row_candidates(
    sources: dict[str, Source],
    row: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Объединяет кандидатов всех источников для одного контекста."""
    rows = [
        (
            sources[name].indices[row],
            sources[name].scores[row],
        )
        for name in SOURCE_NAMES
    ]

    candidates = np.unique(
        np.concatenate([indices[indices >= 0] for indices, _ in rows])
    )

    return candidates, rows
