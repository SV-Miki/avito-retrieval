import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.lexical import (
    LexicalConfig,
    LexicalIndex,
    bm25_weights,
    top_k_indices,
)


def test_top_k_matches_full_sort_with_ties():
    random = np.random.default_rng(42)
    indices = random.permutation(100)
    scores = random.integers(0, 6, size=100).astype(np.float32)

    score_by_index = dict(zip(indices, scores, strict=False))

    expected = sorted(
        (index for index, score in zip(indices, scores, strict=False) if score > 0),
        key=lambda index: (-score_by_index[index], index),
    )

    for k in (1, 10, 50, 120):
        actual = top_k_indices(indices, scores, k)
        assert actual.tolist() == expected[:k]


def test_tfidf_matches_dense_reference_on_small_fixture():
    index = LexicalIndex(LexicalConfig()).fit(
        ['c', 'a', 'b'],
        [
            'красный велосипед',
            'синий велосипед',
            'книга',
        ],
    )

    queries = [
        'велосипед',
        'книга',
        'неизвестно',
        '',
        'велосипед',
    ]

    rankings, _ = index.search(
        queries,
        k=5,
        batch_size=2,
    )

    vectors = index.vectorizer.transform(queries)

    # Плотное представление используется только
    # для небольшой тестовой матрицы.
    scores = cosine_similarity(
        vectors,
        index.postings.T,
    )

    for row, actual in zip(scores, rankings, strict=False):
        expected = sorted(
            np.flatnonzero(row > 0),
            key=lambda item: (-row[item], item),
        )[:5]

        assert actual[actual >= 0].tolist() == expected

    assert rankings[0].tolist() == rankings[-1].tolist()
    assert np.all(rankings[2:4] == -1)


def test_bm25_matches_formula():
    counts = sparse.csr_matrix(
        [
            [2, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ]
    )

    actual = bm25_weights(counts).toarray()

    lengths = np.array([3, 2, 0])
    document_frequency = np.array([1, 2, 1])
    expected = np.zeros((3, 3))

    for row in range(3):
        for column in range(3):
            count = counts[row, column]

            idf = np.log1p(
                (3 - document_frequency[column] + 0.5)
                / (document_frequency[column] + 0.5)
            )

            denominator = count + 1.2 * (
                1 - 0.75 + 0.75 * lengths[row] / lengths.mean()
            )

            expected[row, column] = idf * count * 2.2 / denominator

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    'model',
    ['tfidf', 'bm25'],
)
def test_batch_and_input_order_invariance(model):
    config = LexicalConfig(model=model)

    index = LexicalIndex(config).fit(
        ['b', 'a', 'c'],
        ['кот', 'кот', 'собака'],
    )

    other = LexicalIndex(config).fit(
        ['c', 'a', 'b'],
        ['собака', 'кот', 'кот'],
    )

    queries = [
        'кот',
        'собака',
        'кот кот',
        'нет',
    ]

    first, _ = index.search(
        queries,
        k=2,
        batch_size=1,
    )
    second, _ = other.search(
        queries,
        k=2,
        batch_size=3,
    )

    np.testing.assert_array_equal(first, second)
    assert index.item_ids[first[0]].tolist() == ['a', 'b']


def test_char_partial_match():
    index = LexicalIndex(
        LexicalConfig(
            analyzer='char_wb',
            ngram_range=(3, 5),
        )
    ).fit(
        ['a', 'b'],
        ['велосипед', 'книга'],
    )

    result, _ = index.search(
        ['велосипеды'],
        k=1,
    )

    assert index.item_ids[result[0]].tolist() == ['a']


def test_duplicate_items_rejected():
    with pytest.raises(ValueError):
        LexicalIndex(LexicalConfig()).fit(
            ['a', 'a'],
            ['кот', 'пёс'],
        )
