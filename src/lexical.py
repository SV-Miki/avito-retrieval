"""Лексический поиск по объявлениям с использованием TF-IDF и BM25."""

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def top_k_indices(
    indices: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """Выбирает top-K, разрешая равные оценки по индексу объявления."""
    if k <= 0:
        raise ValueError('k must be positive')

    positive = scores > 0
    indices = indices[positive]
    scores = scores[positive]

    if len(scores) > k:
        threshold = np.partition(scores, len(scores) - k)[-k]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        needed = k - len(above)

        if len(tied) > needed:
            selected = np.argpartition(
                indices[tied],
                needed - 1,
            )[:needed]
            tied = tied[selected]

        keep = np.concatenate([above, tied])
        indices = indices[keep]
        scores = scores[keep]

    order = np.lexsort((indices, -scores))
    return indices[order][:k]


def bm25_weights(
    counts: sparse.csr_matrix,
    k1: float = 1.2,
    b: float = 0.75,
) -> sparse.csr_matrix:
    """Вычисляет веса BM25 для разреженной матрицы документов."""
    matrix = counts.astype(np.float32, copy=True)
    lengths = np.asarray(matrix.sum(axis=1)).ravel()
    average_length = lengths.mean()

    if average_length == 0:
        return matrix

    document_frequency = np.bincount(
        matrix.indices,
        minlength=matrix.shape[1],
    )
    idf = np.log1p(
        (matrix.shape[0] - document_frequency + 0.5) / (document_frequency + 0.5)
    ).astype(np.float32)

    # Обработка блоками ограничивает размер промежуточных массивов.
    for start in range(0, matrix.shape[0], 4096):
        stop = min(start + 4096, matrix.shape[0])
        begin = matrix.indptr[start]
        end = matrix.indptr[stop]

        normalization = k1 * (1 - b + b * lengths[start:stop] / average_length)
        denominator = np.repeat(
            normalization,
            np.diff(matrix.indptr[start : stop + 1]),
        )

        frequencies = matrix.data[begin:end]
        weights = frequencies * (k1 + 1) / (frequencies + denominator)
        matrix.data[begin:end] = weights * idf[matrix.indices[begin:end]]

    return matrix


@dataclass(frozen=True)
class LexicalConfig:
    """Параметры лексического поискового индекса."""

    model: str = 'tfidf'
    analyzer: str = 'word'
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int = 1
    k1: float = 1.2
    b: float = 0.75


class LexicalIndex:
    """Лексический индекс для поиска по текстовым полям объявлений."""

    def __init__(self, config: LexicalConfig):
        self.config = config
        self.vectorizer = None
        self.postings = None
        self.item_ids = None
        self.fit_seconds = 0.0
        self.matrix_shape = None

    def fit(
        self,
        item_ids: Sequence[str],
        texts: Sequence[str],
    ) -> 'LexicalIndex':
        """Строит TF-IDF или BM25 индекс по переданным документам."""
        if len(item_ids) != len(texts) or len(set(item_ids)) != len(item_ids):
            raise ValueError('Expected one document per unique item_id')

        if self.config.model not in ('tfidf', 'bm25'):
            raise ValueError('Unknown lexical model')

        if self.config.model == 'bm25' and self.config.analyzer != 'word':
            raise ValueError('BM25 supports word analyzer only')

        start = perf_counter()

        # Сортировка обеспечивает детерминированное разрешение
        # совпадающих оценок.
        order = np.argsort(
            np.asarray(item_ids),
            kind='stable',
        )
        self.item_ids = np.asarray(item_ids)[order]
        ordered_texts = [texts[index] for index in order]

        options = {
            'analyzer': self.config.analyzer,
            'ngram_range': self.config.ngram_range,
            'min_df': self.config.min_df,
            'lowercase': True,
            'dtype': np.float32,
        }

        if self.config.model == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                norm='l2',
                smooth_idf=True,
                sublinear_tf=False,
                **options,
            )
            matrix = self.vectorizer.fit_transform(ordered_texts)
        else:
            self.vectorizer = CountVectorizer(**options)
            counts = self.vectorizer.fit_transform(ordered_texts)
            matrix = bm25_weights(
                counts,
                self.config.k1,
                self.config.b,
            )

        self.matrix_shape = matrix.shape
        self.postings = matrix.T.tocsr()
        self.fit_seconds = perf_counter() - start

        return self

    def search(
        self,
        texts: Sequence[str],
        k: int = 50,
        batch_size: int = 16,
    ) -> tuple[np.ndarray, dict]:
        """Выполняет поиск по запросам небольшими пакетами.

        Одинаковые тексты запросов обрабатываются один раз.
        Значение -1 означает отсутствие найденного документа.
        Результаты с нулевой оценкой в выдачу не добавляются.
        """
        if self.postings is None:
            raise ValueError('Fit the index before search')

        if k <= 0 or batch_size <= 0:
            raise ValueError('k and batch_size must be positive')

        start = perf_counter()

        unique_texts, inverse = np.unique(
            texts,
            return_inverse=True,
        )
        rankings = np.full(
            (len(unique_texts), k),
            -1,
            dtype=np.int32,
        )

        max_score_bytes = 0
        zero_queries = 0

        for begin in range(0, len(unique_texts), batch_size):
            end = min(begin + batch_size, len(unique_texts))
            queries = self.vectorizer.transform(
                unique_texts[begin:end],
            )

            if self.config.model == 'bm25':
                # Для BM25 запрос представляется набором уникальных терминов.
                queries.data.fill(1)

            zero_queries += int((np.diff(queries.indptr) == 0).sum())

            scores = (queries @ self.postings).tocsr()
            max_score_bytes = max(
                max_score_bytes,
                sparse_bytes(scores),
            )

            for row in range(end - begin):
                left, right = scores.indptr[row : row + 2]
                top = top_k_indices(
                    scores.indices[left:right],
                    scores.data[left:right],
                    k,
                )
                rankings[begin + row, : len(top)] = top

        return rankings[inverse], {
            'retrieval_seconds': perf_counter() - start,
            'unique_query_representations': len(unique_texts),
            'zero_vector_unique_queries': zero_queries,
            'max_batch_score_bytes': max_score_bytes,
            'batch_size': batch_size,
        }

    def statistics(self) -> dict:
        """Возвращает основные характеристики построенного индекса."""
        return {
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'matrix_shape': list(self.matrix_shape),
            'matrix_nnz': self.postings.nnz,
            'index_sparse_bytes': sparse_bytes(self.postings),
            'fit_index_seconds': self.fit_seconds,
        }


def sparse_bytes(matrix: sparse.csr_matrix) -> int:
    """Возвращает объём памяти CSR-матрицы в байтах."""
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
