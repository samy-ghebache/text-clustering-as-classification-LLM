from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "output"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"


@dataclass
class PipelineConfig:
    dataset: str = "arxiv_fine"
    use_large: bool = False
    chunk_size: int = 15
    seed_fraction: float = 0.2
    max_chunks: int | None = None      # limit label generation chunks (None = all)
    max_samples: int | None = None     # limit classification samples (None = all)
    verbose: bool = False
    datasets_dir: Path = DATASETS_DIR
    output_dir: Path = OUTPUT_DIR

    @property
    def size_label(self) -> str:
        return "large" if self.use_large else "small"

    @property
    def data_file(self) -> Path:
        filename = "large.jsonl" if self.use_large else "small.jsonl"
        return self.datasets_dir / self.dataset / filename

    def output_file(self, suffix: str) -> Path:
        return self.output_dir / f"{self.dataset}_{self.size_label}_{suffix}.json"
