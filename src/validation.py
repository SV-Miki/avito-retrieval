"""Подготовка валидационных разбиений и расчёт метрик качества."""

import hashlib
import json
from collections.abc import Collection, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

CONTEXT_COLUMNS = [
    'search_query',
    'search_location_id',
    'search_is_delivery_search',
    'search_infm_params_text',
    'search_category',
]
PARTS = ('train', 'dev', 'holdout')


def normalize_query(text: str | None) -> str | None:
    """Нормализует текст запроса для группового разбиения выборки."""
    if pd.isna(text):
        return None

    return ' '.join(text.lower().replace('ё', 'е').split())


def _json(value: Any) -> str:
    """Сериализует значение в стабильное JSON-представление."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(',', ':'),
        allow_nan=False,
    )


def _scalar(value: Any) -> Any:
    """Преобразует значения NumPy и пропуски в обычные Python-объекты."""
    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


def context_key(values: Iterable[Any]) -> str:
    """Создаёт стабильный идентификатор поискового контекста."""
    values = list(values)

    if len(values) != len(CONTEXT_COLUMNS):
        raise ValueError('Expected all five context fields in CONTEXT_COLUMNS order')

    serialized = _json([_scalar(value) for value in values])

    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class ValidationData:
    """Данные, подготовленные для локальной валидации."""

    contexts: pd.DataFrame
    relevant: dict[str, frozenset[Hashable]]
    corpus: frozenset[Hashable]


def prepare_data(frame: pd.DataFrame) -> ValidationData:
    """Собирает контексты и множества релевантных объявлений."""
    frame = frame[CONTEXT_COLUMNS + ['item_id']]

    if frame.item_id.isna().any():
        raise ValueError('Null item_id cannot be a positive or corpus document')

    contexts = frame[CONTEXT_COLUMNS].drop_duplicates().copy()

    contexts['context_id'] = [
        context_key(row)
        for row in contexts.itertuples(
            index=False,
            name=None,
        )
    ]

    if contexts.context_id.duplicated().any():
        raise ValueError('Context key collision')

    pairs = frame.drop_duplicates().merge(
        contexts,
        on=CONTEXT_COLUMNS,
        validate='many_to_one',
    )

    relevant = (
        pairs.groupby(
            'context_id',
            sort=True,
        )
        .item_id.agg(frozenset)
        .to_dict()
    )

    contexts = contexts.set_index('context_id').sort_index()

    return ValidationData(
        contexts=contexts,
        relevant=relevant,
        corpus=frozenset(frame.item_id),
    )


def _assign(
    keys: Iterable[str],
    seed: int,
) -> dict[str, int]:
    """Детерминированно распределяет группы между частями выборки."""
    # Хеш-сортировка не зависит от исходного порядка строк.
    keys = sorted(
        set(keys),
        key=lambda key: (
            hashlib.sha256(_json([seed, key]).encode()).digest(),
            key,
        ),
    )

    train_end = int(0.8 * len(keys))
    dev_end = int(0.9 * len(keys))

    assignments: dict[str, int] = {}

    for index, key in enumerate(keys):
        if index < train_end:
            assignments[key] = 0
        elif index < dev_end:
            assignments[key] = 1
        else:
            assignments[key] = 2

    return assignments


def make_splits(
    data: ValidationData,
    seed: int = 42,
) -> dict[str, pd.Series]:
    """Создаёт primary и secondary разбиения train/dev/holdout."""
    groups = data.contexts.search_query.map(normalize_query).map(_json)

    primary = _assign(
        groups,
        seed,
    )
    secondary = _assign(
        data.contexts.index,
        seed,
    )

    return {
        'primary': pd.Series(
            [primary[key] for key in groups],
            index=data.contexts.index,
            dtype='uint8',
        ),
        'secondary': pd.Series(
            [secondary[key] for key in data.contexts.index],
            index=data.contexts.index,
            dtype='uint8',
        ),
    }


def labels_for(
    data: ValidationData,
    split: pd.Series,
    part: str = 'train',
) -> dict[str, frozenset[Hashable]]:
    """Возвращает релевантные объявления для выбранной части split."""
    if part not in PARTS:
        raise ValueError(f'Unknown part: {part}')

    part_index = PARTS.index(part)
    context_ids = split.index[split.eq(part_index)]

    return {key: data.relevant[key] for key in context_ids}


def recall_at_k(
    relevant: Iterable[Hashable],
    predictions: Iterable[Hashable],
    k: int = 50,
) -> float:
    """Вычисляет Recall по первым K уникальным предсказаниям."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError('K must be a positive integer')

    relevant_items = set(relevant)

    if not relevant_items:
        raise ValueError('An evaluation context must have observed positives')

    predicted_items = set()

    for item in predictions:
        predicted_items.add(item)

        if len(predicted_items) == k:
            break

    return len(predicted_items & relevant_items) / len(relevant_items)


def evaluate(
    ground_truth: Mapping[str, Collection[Hashable]],
    predictions: Mapping[str, Iterable[Hashable]],
    ks: Iterable[int] = (10, 20, 50),
) -> dict[str, float]:
    """Вычисляет средний Recall@K по поисковым контекстам."""
    if not ground_truth:
        raise ValueError('Empty evaluation set')

    unknown_contexts = set(predictions) - set(ground_truth)

    if unknown_contexts:
        raise ValueError('Predictions contain contexts outside the evaluation set')

    ranked = {key: list(value) for key, value in predictions.items()}

    metrics: dict[str, float] = {}

    for k in ks:
        recalls = [
            recall_at_k(
                items,
                ranked.get(key, ()),
                k,
            )
            for key, items in ground_truth.items()
        ]

        metrics[f'Recall@{k}'] = float(np.mean(recalls))

    return metrics


def _part_statistics(
    labels: Mapping[str, frozenset[Hashable]],
    group_count: int,
    context_count: int,
    missing: set[Hashable],
    part: str,
) -> dict[str, Any]:
    """Собирает статистику для одной части разбиения."""
    counts = np.array([len(items) for items in labels.values()])

    return {
        'normalized_query_groups': group_count,
        'full_query_contexts': context_count,
        'evaluation_contexts': (context_count if part != 'train' else 0),
        'mean_relevant_items': (float(counts.mean()) if len(counts) else None),
        'median_relevant_items': (float(np.median(counts)) if len(counts) else None),
        'contexts_1_relevant': int((counts == 1).sum()),
        'contexts_2plus_relevant': int((counts >= 2).sum()),
        'contexts_over50_relevant': int((counts > 50).sum()),
        'all_positives_in_corpus': not missing,
        'missing_positive_items': len(missing),
        'oracle_recall50_ceiling': (
            float(
                np.minimum(
                    50 / counts,
                    1,
                ).mean()
            )
            if len(counts)
            else None
        ),
    }


def sanity_report(
    data: ValidationData,
    splits: Mapping[str, pd.Series],
) -> dict[str, Any]:
    """Проверяет корректность разбиений и собирает их статистику."""
    report: dict[str, Any] = {
        'source_contexts': len(data.contexts),
        'retrieval_corpus_items': len(data.corpus),
    }

    for scheme, split in splits.items():
        if (
            not split.index.equals(data.contexts.index)
            or not split.isin([0, 1, 2]).all()
        ):
            raise ValueError('Invalid split coverage')

        parts = {}
        group_sets = {}
        context_sets = {}

        train_texts = set(
            data.contexts.loc[
                split.eq(0),
                'search_query',
            ].map(_json)
        )

        for part_index, part in enumerate(PARTS):
            contexts = data.contexts.loc[split.eq(part_index)]

            labels = labels_for(
                data,
                split,
                part,
            )

            groups = set(contexts.search_query.map(normalize_query).map(_json))

            group_sets[part] = groups
            context_sets[part] = set(contexts.index)

            missing: set[Hashable] = set()

            if labels:
                missing = set().union(*labels.values()) - data.corpus

            parts[part] = _part_statistics(
                labels=labels,
                group_count=len(groups),
                context_count=len(contexts),
                missing=missing,
                part=part,
            )

            if part != 'train':
                seen_fraction = (
                    contexts.search_query.map(_json).isin(train_texts).mean()
                )

                parts[part]['exact_query_seen_in_train_fraction'] = float(seen_fraction)

            if missing:
                raise ValueError('Positives absent from corpus')

        pairs = [
            ('train', 'dev'),
            ('train', 'holdout'),
            ('dev', 'holdout'),
        ]

        context_overlap = {
            f'{left}/{right}': len(context_sets[left] & context_sets[right])
            for left, right in pairs
        }

        group_overlap = {
            f'{left}/{right}': len(group_sets[left] & group_sets[right])
            for left, right in pairs
        }

        if any(context_overlap.values()) or (
            scheme == 'primary' and any(group_overlap.values())
        ):
            raise ValueError('Split leakage')

        report[scheme] = {
            'parts': parts,
            'context_overlap': context_overlap,
            'normalized_query_group_overlap': group_overlap,
            'evaluation_contexts_total': int(split.ne(0).sum()),
        }

    return report
