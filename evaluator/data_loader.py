"""
GeLaCo Prototype — Calibration Data Loader
============================================
Loads sentences from English Wikipedia for calibration inference.
The paper uses 64 sentences randomly sampled from wikimedia/wikipedia.
"""

import random

from datasets import load_dataset

from config import (
    DATASET_NAME,
    DATASET_CONFIG,
    NUM_CALIBRATION_SENTENCES,
    RANDOM_SEED,
)


def load_calibration_sentences(
    num_sentences: int = NUM_CALIBRATION_SENTENCES,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """
    Load calibration sentences from English Wikipedia.

    Strategy:
    - Load a streaming subset of Wikipedia articles
    - Extract sentences by splitting article text on '.'
    - Filter for sentences with reasonable length (>50 chars)
    - Sample `num_sentences` with a fixed random seed

    Args:
        num_sentences: Number of sentences to return.
        seed: Random seed for reproducibility.

    Returns:
        List of sentence strings.
    """
    print(f"[DataLoader] Loading Wikipedia dataset: {DATASET_NAME}/{DATASET_CONFIG}")
    print(f"[DataLoader] Target: {num_sentences} calibration sentences (seed={seed})")

    # Use streaming to avoid downloading the entire dataset
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    # Collect candidate sentences from articles
    rng = random.Random(seed)
    candidates = []
    articles_processed = 0
    max_articles = 500  # Process enough articles to get diverse sentences

    for article in dataset:
        if articles_processed >= max_articles:
            break

        text = article.get("text", "")
        if not text:
            continue

        # Split into sentences (simple splitting on period + space)
        raw_sentences = text.split(". ")

        for sent in raw_sentences:
            sent = sent.strip()
            # Filter: reasonable length, not a title/header, has actual content
            if (
                len(sent) > 50
                and len(sent) < 500
                and "\n" not in sent
                and not sent.startswith("==")
                and " " in sent  # Must have multiple words
            ):
                candidates.append(sent + ".")

        articles_processed += 1

    print(f"[DataLoader] Collected {len(candidates)} candidate sentences "
          f"from {articles_processed} articles")

    # Sample fixed number of sentences
    if len(candidates) < num_sentences:
        print(f"[DataLoader] WARNING: Only found {len(candidates)} candidates, "
              f"need {num_sentences}")
        selected = candidates
    else:
        selected = rng.sample(candidates, num_sentences)

    print(f"[DataLoader] Selected {len(selected)} calibration sentences")
    for i, sent in enumerate(selected):
        preview = sent[:80] + "..." if len(sent) > 80 else sent
        print(f"  [{i}] {preview}")

    # Explicitly delete the dataset iterator and run GC to prevent PyArrow
    # C++ threads from crashing when the Python GIL is released on shutdown.
    del dataset
    import gc
    gc.collect()

    return selected


if __name__ == "__main__":
    sentences = load_calibration_sentences()
    print(f"\nTotal sentences loaded: {len(sentences)}")
    
    # Force exit to prevent PyArrow background thread segfaults during 
    # Python runtime finalization when this script is run standalone.
    import os
    os._exit(0)
