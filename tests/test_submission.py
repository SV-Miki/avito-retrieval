from pathlib import Path

import pandas as pd

from src.submission import audit_submission


def test_audit_accepts_literal_ids_and_csv_round_trip(
    tmp_path: Path,
) -> None:
    queries = pd.DataFrame(
        {
            'query_id': ['0123456789abcdef'],
        }
    )
    items = pd.DataFrame(
        {
            'item_id': ['fedcba9876543210'],
        }
    )
    answer = pd.DataFrame(
        {
            'query_id': ['0123456789abcdef'],
            'answer': ['fedcba9876543210'],
        }
    )

    output = tmp_path / 'answer.csv'
    answer.to_csv(output, index=False, encoding='utf-8')

    audit_submission(
        answer=answer,
        queries=queries,
        items=items,
        output=output,
    )
