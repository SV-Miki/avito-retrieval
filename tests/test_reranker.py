import numpy as np
import pandas as pd

from src.reranker import build_training_data, candidate_matrix
from src.retrieval import SOURCE_NAMES, Source


def make_sources() -> dict[str, Source]:
    return {
        name: Source(
            name=name,
            indices=np.array(
                [[0, 1, 2, -1]],
                dtype=np.int32,
            ),
            scores=np.array(
                [[3.0, 2.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
        )
        for name in SOURCE_NAMES
    }


def test_candidate_features_shape_and_location():
    rows = [
        (
            source.indices[0],
            source.scores[0],
        )
        for source in make_sources().values()
    ]

    matrix = candidate_matrix(
        source_rows=rows,
        candidates=np.array([0, 1, 2]),
        search_location=10,
        item_locations=np.array([10, 20, 10]),
        query_length=4,
        title_lengths=np.array([5, 6, 7]),
    )

    assert matrix.shape == (3, 19)
    assert matrix[:, 15].tolist() == [5.0, 5.0, 5.0]
    assert matrix[:, 16].tolist() == [1.0, 0.0, 1.0]


def test_negative_sampling_is_deterministic():
    contexts = pd.DataFrame(
        {
            'search_location_id': [10],
            'search_query': ['query'],
        },
        index=['ctx'],
    )

    arguments = (
        contexts,
        {'ctx': frozenset({'item-0'})},
        make_sources(),
        {'item-0': 0},
        np.array([10, 20, 10]),
        np.array([5, 6, 7]),
    )

    first = build_training_data(*arguments)
    second = build_training_data(*arguments)

    np.testing.assert_array_equal(
        first[0],
        second[0],
    )
    np.testing.assert_array_equal(
        first[1],
        second[1],
    )
