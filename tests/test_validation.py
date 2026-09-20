import pandas as pd
import pytest

from src.validation import (
    CONTEXT_COLUMNS,
    PARTS,
    context_key,
    evaluate,
    labels_for,
    make_splits,
    normalize_query,
    prepare_data,
    recall_at_k,
    sanity_report,
)

SPLIT_SCHEMES = ('primary', 'secondary')


def make_frame() -> pd.DataFrame:
    rows = []

    for index in range(100):
        for location in (1, 2, 3):
            query = f' Ёлка {index} ' if location == 1 else f'елка {index}'

            for item in (index, index + 100):
                rows.append([query, location, False, None, 7, item])

    return pd.DataFrame(rows, columns=CONTEXT_COLUMNS + ['item_id'])


class TestRecall:
    def test_single_found_and_missing(self):
        assert recall_at_k([1], [1]) == 1
        assert recall_at_k([1], [2]) == 0

    def test_multiple_relevant(self):
        assert recall_at_k([1, 2, 3], [1, 3, 9]) == pytest.approx(2 / 3)
        assert recall_at_k([1, 1, 2], [1]) == 0.5

    def test_duplicate_predictions(self):
        assert recall_at_k([1, 2], [1, 1, 1, 2], 2) == 1
        assert recall_at_k([1, 2], [1, 1], 50) == 0.5

    def test_short_empty_and_cutoff(self):
        assert recall_at_k([1, 2], [2], 50) == 0.5
        assert recall_at_k([1], [], 50) == 0
        assert recall_at_k([51], range(1, 52), 50) == 0
        assert recall_at_k(range(100), range(100), 50) == 0.5

    def test_macro_and_missing_contexts(self):
        truth = {
            'a': {1},
            'b': {2, 3, 4},
            'c': {5},
        }
        result = evaluate(truth, {'a': [1], 'b': [2]})

        assert set(result) == {'Recall@10', 'Recall@20', 'Recall@50'}
        assert result['Recall@50'] == pytest.approx((1 + 1 / 3 + 0) / 3)

    def test_invalid_inputs(self):
        for k in (0, -1, 1.5, True):
            with pytest.raises(ValueError):
                recall_at_k([1], [1], k)

        with pytest.raises(ValueError):
            recall_at_k([], [1])

        with pytest.raises(ValueError):
            evaluate({}, {})

        with pytest.raises(ValueError):
            evaluate({'a': {1}}, {'wrong': [1]})

    def test_generator_predictions(self):
        result = evaluate({'a': {1}}, {'a': iter([1])})

        assert result == {
            'Recall@10': 1,
            'Recall@20': 1,
            'Recall@50': 1,
        }


class TestSplit:
    @pytest.fixture(autouse=True)
    def validation_data(self):
        self.frame = make_frame()
        self.data = prepare_data(self.frame)
        self.splits = make_splits(self.data, seed=42)

    def test_normalization_only_split_key(self):
        assert normalize_query('  ЁЛКА\t \nдом  ') == 'елка дом'

        original_key = context_key(['Ёлка', 1, False, None, 7])
        normalized_key = context_key(['елка', 1, False, None, 7])
        assert original_key != normalized_key

        null_key = context_key(['a', 1, False, None, 7])
        empty_key = context_key(['a', 1, False, '', 7])
        nan_key = context_key(['a', 1, False, float('nan'), 7])

        assert null_key != empty_key
        assert null_key == nan_key

    def test_disjoint_contexts_and_primary_groups(self):
        for scheme in SPLIT_SCHEMES:
            split = self.splits[scheme]

            context_sets = [
                set(split.index[split.eq(part_index)]) for part_index in range(3)
            ]

            assert set.union(*context_sets) == set(self.data.contexts.index)

            for left, right in ((0, 1), (0, 2), (1, 2)):
                assert not context_sets[left] & context_sets[right]

            if scheme == 'primary':
                groups = []

                for part_index in range(3):
                    queries = self.data.contexts.loc[
                        split.eq(part_index),
                        'search_query',
                    ]
                    groups.append(set(queries.map(normalize_query)))

                assert [len(group) for group in groups] == [80, 10, 10]

                for left, right in ((0, 1), (0, 2), (1, 2)):
                    assert not groups[left] & groups[right]

        counts = self.splits['secondary'].value_counts().sort_index()
        assert counts.tolist() == [240, 30, 30]

    def test_reproducible_independent_of_row_order_and_duplicates(self):
        shuffled = pd.concat(
            [
                self.frame,
                self.frame.iloc[:100],
            ]
        ).sample(
            frac=1,
            random_state=123,
        )

        other = prepare_data(shuffled)

        assert other.relevant == self.data.relevant
        assert other.corpus == self.data.corpus

        for scheme, split in make_splits(other, seed=42).items():
            pd.testing.assert_series_equal(split, self.splits[scheme])

        other_seed_split = make_splits(other, seed=43)['primary']
        assert not other_seed_split.equals(self.splits['primary'])

    def test_context_fields_and_ground_truth_deduplication(self):
        base = ['query', 1, False, None, 7]
        rows = [
            base + [1],
            base + [1],
            base + [2],
        ]

        for field, value in enumerate(['QUERY', 2, True, 'filter', 8]):
            changed = base.copy()
            changed[field] = value
            rows.append(changed + [3])

        frame = pd.DataFrame(
            rows,
            columns=CONTEXT_COLUMNS + ['item_id'],
        )
        data = prepare_data(frame)

        assert len(data.contexts) == 6
        assert data.relevant[context_key(base)] == {1, 2}

    def test_training_labels_and_corpus(self):
        for scheme in SPLIT_SCHEMES:
            labels = [
                labels_for(self.data, self.splits[scheme], part) for part in PARTS
            ]

            assert not set(labels[0]) & (set(labels[1]) | set(labels[2]))

            for part_labels in labels:
                for positives in part_labels.values():
                    assert positives <= self.data.corpus

        assert self.data.corpus == frozenset(self.frame.item_id)

        train_labels = labels_for(self.data, self.splits['primary'])
        dev_labels = labels_for(self.data, self.splits['primary'], 'dev')

        train_items = set().union(*train_labels.values())
        dev_items = set().union(*dev_labels.values())

        assert dev_items - train_items
        assert dev_items <= self.data.corpus

    def test_sanity_and_secondary_familiar_queries(self):
        report = sanity_report(self.data, self.splits)

        assert report['retrieval_corpus_items'] == 200
        assert report['primary']['evaluation_contexts_total'] == 60

        secondary_dev = report['secondary']['parts']['dev']
        primary_dev = report['primary']['parts']['dev']

        assert secondary_dev['exact_query_seen_in_train_fraction'] > 0
        assert primary_dev['exact_query_seen_in_train_fraction'] == 0
