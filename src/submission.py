"""Генерация и проверка итогового benchmark submission."""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.reranker import build_training_data, fit_reranker, predict_top50
from src.retrieval import field_text, retrieve_sources
from src.semantic import encode_passages, load_encoder
from src.validation import (
    CONTEXT_COLUMNS,
    labels_for,
    make_splits,
    prepare_data,
)

ITEM_PATTERN = re.compile(r'^[0-9a-f]{16}$')

TRAIN_PATH = Path('data/train.parquet')
BENCHMARK_QUERIES_PATH = Path('data/benchmark_queries.parquet')
BENCHMARK_ITEMS_PATH = Path('data/benchmark_items.parquet')

ITEM_COLUMNS = [
    'item_id',
    'item_title_raw',
    'item_infm_params_text',
    'item_description_raw',
    'item_location_id',
]


def audit_submission(
    answer: pd.DataFrame,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    output: Path,
) -> None:
    """Проверяет формат итогового файла и корректность записанных данных."""
    expected_query_ids = queries.query_id.astype(str).tolist()
    item_ids = set(items.item_id.astype(str))

    if answer.columns.tolist() != ['query_id', 'answer']:
        raise ValueError('Invalid submission columns')

    if answer.query_id.tolist() != expected_query_ids:
        raise ValueError('Invalid query_id order or values')

    if len(answer) != len(queries):
        raise ValueError('Invalid query count')

    if not answer.query_id.is_unique:
        raise ValueError('Duplicate query_id values')

    for value in answer.answer:
        if not value:
            raise ValueError('Empty answer')

        tokens = value.split(' ')

        if len(tokens) > 50:
            raise ValueError('Answer contains more than 50 items')

        if len(tokens) != len(set(tokens)):
            raise ValueError('Answer contains duplicate item IDs')

        if ' '.join(tokens) != value:
            raise ValueError('Invalid answer separator')

        valid_items = all(
            ITEM_PATTERN.fullmatch(token) and token in item_ids for token in tokens
        )

        if not valid_items:
            raise ValueError('Invalid answer item IDs')

    loaded = pd.read_csv(
        output,
        dtype={
            'query_id': str,
            'answer': str,
        },
        keep_default_na=False,
    )

    if not loaded.equals(answer):
        raise ValueError('CSV round-trip changed submission')


def main(output: Path) -> None:
    """Обучает reranker и формирует итоговый benchmark submission."""
    labels = pd.read_parquet(
        TRAIN_PATH,
        columns=CONTEXT_COLUMNS + ['item_id'],
    )

    validation_data = prepare_data(labels)
    splits = make_splits(
        validation_data,
        seed=42,
    )

    truth = labels_for(
        validation_data,
        splits['primary'],
        'train',
    )

    contexts = validation_data.contexts.loc[list(truth)]

    train_items = (
        pd.read_parquet(
            TRAIN_PATH,
            columns=ITEM_COLUMNS,
        )
        .drop_duplicates('item_id')
        .sort_values('item_id')
    )

    train_item_ids = train_items.item_id.to_numpy()
    train_locations = train_items.item_location_id.to_numpy()

    train_titles = field_text(
        train_items,
        'item_title_raw',
    )
    train_params = field_text(
        train_items,
        'item_infm_params_text',
    )
    train_descriptions = [
        text[:512]
        for text in field_text(
            train_items,
            'item_description_raw',
        )
    ]
    train_title_lengths = (
        train_items.item_title_raw.fillna('')
        .astype(str)
        .str.len()
        .to_numpy(dtype=np.int32)
    )

    encoder = load_encoder()

    train_embeddings = encode_passages(
        encoder,
        train_titles,
    )

    train_sources = retrieve_sources(
        contexts=contexts,
        item_ids=train_item_ids,
        titles=train_titles,
        params=train_params,
        descriptions=train_descriptions,
        encoder=encoder,
        item_embeddings=train_embeddings,
    )

    item_index = {item_id: index for index, item_id in enumerate(train_item_ids)}

    features, targets = build_training_data(
        contexts=contexts,
        truth=truth,
        sources=train_sources,
        item_index=item_index,
        item_locations=train_locations,
        title_lengths=train_title_lengths,
    )

    scaler, model = fit_reranker(
        features,
        targets,
    )

    del (
        features,
        targets,
        train_sources,
        train_items,
        train_embeddings,
    )
    gc.collect()

    queries = pd.read_parquet(
        BENCHMARK_QUERIES_PATH,
    )
    queries['query_id'] = queries.query_id.astype(str)

    items = pd.read_parquet(
        BENCHMARK_ITEMS_PATH,
    )
    items['item_id'] = items.item_id.astype(str)

    items = items.drop_duplicates('item_id').sort_values('item_id')

    item_ids = items.item_id.to_numpy()
    item_locations = items.item_location_id.to_numpy()

    item_titles = field_text(
        items,
        'item_title_raw',
    )
    item_params = field_text(
        items,
        'item_infm_params_text',
    )
    item_descriptions = [
        text[:512]
        for text in field_text(
            items,
            'item_description_raw',
        )
    ]
    item_title_lengths = (
        items.item_title_raw.fillna('').astype(str).str.len().to_numpy(dtype=np.int32)
    )

    benchmark_embeddings = encode_passages(
        encoder,
        item_titles,
    )

    benchmark_sources = retrieve_sources(
        contexts=queries,
        item_ids=item_ids,
        titles=item_titles,
        params=item_params,
        descriptions=item_descriptions,
        encoder=encoder,
        item_embeddings=benchmark_embeddings,
    )

    ranked = predict_top50(
        contexts=queries,
        sources=benchmark_sources,
        item_locations=item_locations,
        title_lengths=item_title_lengths,
        item_ids=item_ids,
        scaler=scaler,
        model=model,
    )

    answer = pd.DataFrame(
        {
            'query_id': queries.query_id,
            'answer': [' '.join(row) for row in ranked],
        }
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    answer.to_csv(
        output,
        index=False,
        encoding='utf-8',
    )

    audit_submission(
        answer=answer,
        queries=queries,
        items=items,
        output=output,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate benchmark submission.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('answer.csv'),
        help='Path to the output CSV file.',
    )

    arguments = parser.parse_args()

    with threadpool_limits(limits=1):
        main(arguments.output)
