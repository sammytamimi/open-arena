import json
import logging
from collections import defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langfuse import get_client
from tqdm import tqdm

from src.datasets.base import Row

_logger = logging.getLogger(__name__)
_MISSING_PREVIEW_LENGTH = 50


def _ensure_dataset(langfuse, name: str, description: str = "") -> None:
    try:
        langfuse.get_dataset(name)
    except Exception:
        _logger.debug(f"Creating new Langfuse dataset '{name}'")
        langfuse.create_dataset(name=name, description=description)


def upload_rows(
    rows: Iterable[Row],
    dataset_name: str,
    description: str = "",
    max_workers: int = 12,
) -> list[Row]:
    """
    Create/get the Langfuse dataset and upload each row. Returns new rows with
    lf_item_id / lf_dataset_id / lf_dataset_name merged into metadata so the
    execution layer can link traces back to items.
    """
    rows = list(rows)
    if not rows:
        _logger.warning("No rows to upload to Langfuse")
        return rows

    langfuse = get_client()
    _ensure_dataset(langfuse, dataset_name, description)

    def _upload_one(row: Row) -> Row:
        input_, expected, metadata = row
        created = langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=input_,
            expected_output=expected,
            metadata=metadata,
        )
        return (
            input_,
            expected,
            {
                **metadata,
                "lf_item_id": created.id,
                "lf_dataset_id": created.dataset_id,
                "lf_dataset_name": created.dataset_name,
            },
        )

    uploaded: list[Row] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(_upload_one, r): (i, r) for i, r in enumerate(rows, start=1)
        }
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Uploading to Langfuse"):
            index, (input_, _expected, _metadata) = future_to_row[future]
            try:
                uploaded.append(future.result())
            except Exception as e:
                _logger.error(f"Upload failed for row {index}: {e}")
                failures.append(f"row {index} ({input_[:_MISSING_PREVIEW_LENGTH]!r}): {e}")

    if failures:
        preview = "; ".join(failures[:5]) + (", ..." if len(failures) > 5 else "")
        raise RuntimeError(
            f"Failed to upload {len(failures)}/{len(rows)} rows to Langfuse dataset "
            f"'{dataset_name}': {preview}"
        )

    return uploaded


def _item_key(input_value: Any, expected_output: Any, metadata: dict[str, Any] | None) -> str:
    def normalize(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, (list, tuple)):
            return [normalize(x) for x in value]
        return str(value)

    return json.dumps(
        {"input": normalize(input_value), "expected_output": normalize(expected_output), "metadata": normalize(metadata or {})},
        sort_keys=True,
    )


def attach_existing_dataset(rows: Iterable[Row], dataset_name: str) -> list[Row]:
    """
    For the --skip-upload path: match each row against an existing Langfuse
    dataset and annotate metadata with the discovered lf_* ids. Fails fast if
    any row cannot be matched.
    """
    rows = list(rows)
    if not rows:
        return rows

    langfuse = get_client()
    _logger.info(f"Loading existing Langfuse dataset: {dataset_name}")
    remote = langfuse.get_dataset(dataset_name)

    by_key: dict[str, deque] = defaultdict(deque)
    for item in remote.items:
        by_key[_item_key(item.input, item.expected_output, item.metadata)].append(item)

    annotated: list[Row] = []
    missing: list[str] = []
    for index, (input_, expected, metadata) in enumerate(rows, start=1):
        matching = by_key.get(_item_key(input_, expected, metadata))
        if not matching:
            missing.append(f"row {index} ({input_[:_MISSING_PREVIEW_LENGTH]!r})")
            annotated.append((input_, expected, metadata))
            continue
        remote_item = matching.popleft()
        annotated.append((
            input_,
            expected,
            {
                **metadata,
                "lf_item_id": remote_item.id,
                "lf_dataset_id": remote.id,
                "lf_dataset_name": remote.name,
            },
        ))

    if missing:
        preview = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        raise ValueError(
            f"Some rows were not found in existing Langfuse dataset '{dataset_name}': {preview}. "
            "Re-run without --skip-upload to upload again."
        )

    return annotated
