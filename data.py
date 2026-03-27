import json
from pathlib import Path
from .config import PipelineConfig


def load_dataset(config: PipelineConfig) -> list[dict]:
    path = config.data_file
    print(f"Loading dataset: {path}")
    data_list = []
    with open(path, "r") as f:
        for line in f:
            data_list.append(json.loads(line))
    print(f"Loaded {len(data_list)} samples")
    return data_list


def get_label_list(data_list: list[dict]) -> list[str]:
    seen: set[str] = set()
    labels: list[str] = []
    for item in data_list:
        label = item["label"]
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def get_sentences(data_list: list[dict]) -> list[str]:
    return [item["input"] for item in data_list]


def list_datasets(datasets_dir: Path) -> list[str]:
    return sorted(
        p.name for p in datasets_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {path}")


def read_json(path: Path) -> object:
    with open(path, "r") as f:
        return json.load(f)
