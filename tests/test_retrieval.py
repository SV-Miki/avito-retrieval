import numpy as np
import pandas as pd

from src.retrieval import (
    SOURCE_NAMES,
    Source,
    params_queries,
    row_candidates,
)
from src.semantic import top_k_dot_product


def test_semantic_top_k_matches_reference():
    queries = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    items = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    actual, scores = top_k_dot_product(
        queries,
        items,
        k=2,
        batch_size=1,
    )

    expected = np.argsort(
        -(queries @ items.T),
        axis=1,
    )[:, :2]

    np.testing.assert_array_equal(
        actual,
        expected,
    )
    assert scores.shape == (2, 2)


def test_candidate_union_uses_all_sources_once():
    sources = {
        name: Source(
            name=name,
            indices=np.array(
                [[index, -1]],
                dtype=np.int32,
            ),
            scores=np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            ),
        )
        for index, name in enumerate(SOURCE_NAMES)
    }

    candidates, rows = row_candidates(
        sources,
        row=0,
    )

    assert candidates.tolist() == [0, 1, 2, 3, 4]
    assert len(rows) == len(SOURCE_NAMES)


def test_params_query_adds_only_nonempty_filter():
    contexts = pd.DataFrame(
        {
            'search_query': [
                'телефон',
                'велосипед',
            ],
            'search_infm_params_text': [
                '',
                ' красный ',
            ],
        }
    )

    assert params_queries(contexts) == [
        'телефон',
        'велосипед  красный ',
    ]
