import json


def generate_label_prompt(sentences: list[str], given_labels: list[str]) -> str:
    example = json.dumps({"new_labels": ["label name"]})
    labels_json = json.dumps(given_labels)
    return (
        "You are helping to discover meaningful category labels for text clustering.\n"
        "Given the existing labels and the sentences below, determine if any sentences "
        "do not fit the existing labels. If so, generate new meaningful label names "
        "for those sentences.\n\n"
        "Guidelines:\n"
        "- First check if a sentence fits an existing label before creating a new one.\n"
        "- Do NOT create a label that means the same thing as an existing one "
        "(e.g. do not create 'appreciation' if 'gratitude' already exists).\n"
        "- Keep new labels short and at the same level of generality "
        "as the existing labels.\n"
        "- Only create labels for essential, distinct clusters.\n"
        "- If all sentences fit existing labels, return an empty list.\n\n"
        f"Existing labels: {labels_json}\n"
        f"Sentences: {json.dumps(sentences)}\n\n"
        f"Return ONLY the new labels as JSON: {example}\n"
        'If no new labels are needed: {"new_labels": []}'
    )


def merge_label_prompt(label_list: list[str]) -> str:
    example = json.dumps({"merged_labels": ["label name", "label name"]})
    return (
        "Please analyze the provided list of labels to identify entries that are similar "
        "or duplicate, considering synonyms, variations in phrasing, and closely related "
        "terms that essentially refer to the same concept. Merge these similar entries into "
        "a single representative label for each unique concept identified. Simplify the list "
        "by reducing redundancies without organizing it into subcategories.\n"
        f"Labels: {label_list}\n"
        f"Return the simplified list as flat JSON like: {example}"
    )


def classify_sentence_prompt(label_list: list[str], sentence: str) -> str:
    example = json.dumps({"label": "chosen label"})
    labels_json = json.dumps(label_list)
    return (
        "Pick the single best label for the sentence from the list below.\n"
        "You MUST pick exactly one label from the list. Pick the closest match "
        "even if it is not perfect. Never say 'unknown', 'none', or 'other'.\n\n"
        f"Labels: {labels_json}\n"
        f"Sentence: {sentence}\n\n"
        f"Return ONLY valid JSON: {example}"
    )
