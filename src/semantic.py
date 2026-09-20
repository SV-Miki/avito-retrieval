"""Семантический поиск по объявлениям с использованием multilingual-e5-small."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'intfloat/multilingual-e5-small'
MAX_SEQUENCE_LENGTH = 512
ENCODING_BATCH_SIZE = 64


def load_encoder() -> SentenceTransformer:
    """Загружает E5-encoder и выбирает доступное устройство выполнения."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    encoder = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )
    encoder.max_seq_length = MAX_SEQUENCE_LENGTH

    return encoder


def top_k_dot_product(
    queries: np.ndarray,
    items: np.ndarray,
    k: int,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Выполняет точный top-K поиск по скалярному произведению."""
    if queries.ndim != 2 or items.ndim != 2:
        raise ValueError('queries and items must be two-dimensional')

    if queries.shape[1] != items.shape[1]:
        raise ValueError('queries and items must have the same dimension')

    if k <= 0 or batch_size <= 0:
        raise ValueError('k and batch_size must be positive')

    if not len(items):
        raise ValueError('items must not be empty')

    k = min(k, len(items))

    result_ids = np.empty(
        (len(queries), k),
        dtype=np.int32,
    )
    result_scores = np.empty(
        (len(queries), k),
        dtype=np.float32,
    )

    for start in range(0, len(queries), batch_size):
        stop = min(
            start + batch_size,
            len(queries),
        )

        scores = queries[start:stop] @ items.T

        candidates = np.argpartition(
            scores,
            -k,
            axis=1,
        )[:, -k:]

        candidate_scores = np.take_along_axis(
            scores,
            candidates,
            axis=1,
        )

        order = np.argsort(
            -candidate_scores,
            axis=1,
            kind='stable',
        )

        result_ids[start:stop] = np.take_along_axis(
            candidates,
            order,
            axis=1,
        )
        result_scores[start:stop] = np.take_along_axis(
            candidate_scores,
            order,
            axis=1,
        )

    return result_ids, result_scores


def encode_passages(
    encoder: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    """Кодирует заголовки объявлений в passage-формате модели E5."""
    passages = [f'passage: {text}' for text in texts]

    return encoder.encode(
        passages,
        batch_size=ENCODING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def encode_unique_queries(
    encoder: SentenceTransformer,
    texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Кодирует уникальные запросы и сохраняет их исходный порядок."""
    unique_texts, inverse = np.unique(
        np.asarray(texts),
        return_inverse=True,
    )

    queries = [f'query: {text}' for text in unique_texts]

    embeddings = encoder.encode(
        queries,
        batch_size=ENCODING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    return embeddings, inverse
