import math
import random
import time
from datetime import datetime
from tqdm import tqdm
from .config import PipelineConfig, MODEL
from .data import (
    load_dataset, get_label_list, get_sentences, list_datasets,
    write_json, read_json,
)
from .llm import chat_json, chat_json_stream
from .prompts import generate_label_prompt, merge_label_prompt, classify_sentence_prompt
from .evaluate import compute_scores


# -- Step 1: Select seed labels --

def select_seed_labels(config: PipelineConfig) -> dict[str, list[str]]:
    datasets = (
        [config.dataset] if config.dataset != "all"
        else list_datasets(config.datasets_dir)
    )
    chosen: dict[str, list[str]] = {}
    for ds_name in datasets:
        ds_config = PipelineConfig(
            dataset=ds_name,
            datasets_dir=config.datasets_dir,
            output_dir=config.output_dir,
        )
        data_list = load_dataset(ds_config)
        labels = get_label_list(data_list)
        k = max(1, int(config.seed_fraction * len(labels)))
        chosen[ds_name] = random.choices(labels, k=k)
        print(f"  {ds_name}: {len(labels)} labels, chose {k}")

    out_path = config.output_dir / "chosen_labels.json"
    write_json(chosen, out_path)
    return chosen


# -- Step 2: Label generation --

def generate_labels(config: PipelineConfig) -> list[str]:
    data_list = load_dataset(config)
    random.shuffle(data_list)
    true_labels = get_label_list(data_list)
    write_json(true_labels, config.output_file("true_labels"))

    chosen_labels_path = config.output_dir / "chosen_labels.json"
    all_chosen = read_json(chosen_labels_path)
    seed_labels: list[str] = all_chosen[config.dataset]
    seed_labels_norm = [l.lower().strip() for l in seed_labels]

    all_labels = list(seed_labels_norm)

    total_chunks = math.ceil(len(data_list) / config.chunk_size)
    if config.max_chunks:
        total_chunks = min(total_chunks, config.max_chunks)

    chunk_iter = range(0, total_chunks * config.chunk_size, config.chunk_size)
    pbar = tqdm(chunk_iter, total=total_chunks, desc="Label generation", unit="chunk")

    for i in pbar:
        chunk = data_list[i : i + config.chunk_size]
        sentences = get_sentences(chunk)
        # Pass only seed labels (keeps prompt size constant)
        prompt = generate_label_prompt(sentences, seed_labels_norm)
        parsed = chat_json(prompt)

        if parsed is None:
            continue

        if isinstance(parsed, dict):
            first_val = next(iter(parsed.values()), [])
            new_labels = first_val if isinstance(first_val, list) else [first_val]
        else:
            continue

        # Collect valid new labels
        added = 0
        for label in new_labels:
            if isinstance(label, str) and "unknown_topic" not in label and "new_label" not in label:
                norm = label.lower().strip()
                if norm and norm not in all_labels:
                    all_labels.append(norm)
                    added += 1

        pbar.set_postfix(labels=len(all_labels), new=added)

    write_json(all_labels, config.output_file("llm_labels_before_merge"))
    print(f"{len(all_labels)} labels before merge")

    # LLM-based merge (streaming so you can see output live)
    print("Merging similar labels via LLM (streaming)...")
    prompt = merge_label_prompt(all_labels)
    merged_parsed = chat_json_stream(prompt)
    if merged_parsed and isinstance(merged_parsed, dict):
        merged_labels = []
        for val in merged_parsed.values():
            if isinstance(val, list):
                merged_labels.extend(val)
            else:
                merged_labels.append(val)
        merged_labels = [l.lower().strip() for l in merged_labels if isinstance(l, str) and l.strip()]
        merged_labels = list(dict.fromkeys(merged_labels))  # deduplicate
    else:
        merged_labels = all_labels

    write_json(merged_labels, config.output_file("llm_labels_after_merge"))
    print(f"Labels after merge: {len(merged_labels)}")
    return merged_labels


# -- Step 3: Classification --

def _extract_label(response: dict | list | str | None, label_set: set[str]) -> str:
    if response is None:
        return "Unsuccessful"
    # Try exact match from JSON dict
    if isinstance(response, dict):
        for val in response.values():
            if isinstance(val, str) and val.lower().strip() in label_set:
                return val.lower().strip()
    # Substring fallback: search raw string for any known label
    raw = str(response).lower()
    for label in sorted(label_set, key=len, reverse=True):  # longest first to avoid partial matches
        if label in raw:
            return label
    return "Unsuccessful"


def classify_with_labels(config: PipelineConfig) -> list[dict]:
    data_list = load_dataset(config)
    merged_path = config.output_file("llm_labels_after_merge")
    label_list = list({l.lower().strip() for l in read_json(merged_path)})
    label_set = set(label_list)
    print(f"Classifying with {len(label_list)} labels")

    results: list[dict] = []
    limit = config.max_samples if config.max_samples else len(data_list)
    limit = min(limit, len(data_list))

    pbar = tqdm(range(limit), desc="Classification", unit="sample")
    unsuccessful = 0

    for i in pbar:
        sentence = data_list[i]["input"]
        true_label = data_list[i]["label"]
        prompt = classify_sentence_prompt(label_list, sentence)
        parsed = chat_json(prompt)
        pred_label = _extract_label(parsed, label_set)

        # Retry once on failure
        if pred_label == "Unsuccessful":
            parsed = chat_json(prompt)
            pred_label = _extract_label(parsed, label_set)

        if pred_label == "Unsuccessful":
            unsuccessful += 1

        results.append({
            "input": sentence,
            "true_label": true_label,
            "pred_label": pred_label,
        })

        pbar.set_postfix(ok=i + 1 - unsuccessful, fail=unsuccessful)

        # Checkpoint every 200 samples
        if i > 0 and i % 200 == 0:
            write_json(results, config.output_file("classification"))

    write_json(results, config.output_file("classification"))
    print(f"  Total: {len(results)}, Unsuccessful: {unsuccessful}")
    return results


# -- Step 4: Evaluate --

def evaluate(config: PipelineConfig, run_start: float | None = None) -> None:
    results = read_json(config.output_file("classification"))

    matched = [r for r in results if r["pred_label"] != "Unsuccessful"]
    unsuccessful = len(results) - len(matched)
    print(f"Evaluating {len(matched)}/{len(results)} samples (excluded Unsuccessful)")

    if not matched:
        print("No valid predictions to evaluate.")
        return

    true_labels = [r["true_label"] for r in matched]
    pred_labels = [r["pred_label"] for r in matched]

    scores = compute_scores(true_labels, pred_labels)
    print(f"Results for {config.dataset}: {scores}")

    # Save log
    log = {
        "timestamp": datetime.now().isoformat(),
        "dataset": config.dataset,
        "split": config.size_label,
        "model": MODEL,
        "config": {
            "chunk_size": config.chunk_size,
            "seed_fraction": config.seed_fraction,
            "max_chunks": config.max_chunks,
            "max_samples": config.max_samples,
        },
        "results": {
            "total_samples": len(results),
            "matched_samples": len(matched),
            "unsuccessful": unsuccessful,
            "unsuccessful_pct": round(unsuccessful / len(results) * 100, 1),
            "ACC": round(scores.acc, 4),
            "ARI": round(scores.ari, 4),
            "NMI": round(scores.nmi, 4),
        },
        "total_duration_s": round(time.time() - run_start, 1) if run_start else None,
    }
    log_path = config.output_dir / "runs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.dataset}.json"
    write_json(log, log_path)
    print(f"Log saved → {log_path}")


# -- Full pipeline --

def run_full(config: PipelineConfig) -> None:
    run_start = time.time()
    print("=== Step 1: Select seed labels ===")
    select_seed_labels(config)
    print("\n=== Step 2: Generate labels ===")
    generate_labels(config)
    print("\n=== Step 3: Classify sentences ===")
    classify_with_labels(config)
    print("\n=== Step 4: Evaluate ===")
    evaluate(config, run_start=run_start)
