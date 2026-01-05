import json
from pathlib import Path


def append_training_record(path: Path, record: dict, *, validate: bool = False) -> None:
    if validate:
        from .schema import validate_training_record

        ok, errors = validate_training_record(record)
        if not ok:
            raise ValueError(f"Invalid training record: {errors}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
