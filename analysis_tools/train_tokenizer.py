"""Rebuild the SentencePiece tokenizer from its listed public training records.

Granary texts are read from the listed byte blocks of the English ASR manifests
at a fixed revision; TED-LIUM texts come from Rev's linked training manifest.
The training text is checked against its recorded SHA-256, and the trained
model is written to data/tokenizer/tokenizer.model.
"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re

import requests

from analysis_tools.generate_workloads import (
    GRANARY,
    TEDLIUM,
    clean_tedlium,
    json_records,
)

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data/tokenizer/training_records.json"
MODEL = ROOT / "data/tokenizer/tokenizer.model"
BUILD = ROOT / "build/tokenizer"
# The arguments NeMo's process_asr_text_tokenizer.py passes for an SPE-BPE model
# with --no_lower_case and --spe_max_sentencepiece_length 512.
TRAINER = dict(
    vocab_size=1023,
    shuffle_input_sentence=True,
    hard_vocab_limit=False,
    model_type="bpe",
    character_coverage=1.0,
    bos_id=-1,
    eos_id=-1,
    max_sentencepiece_length=512,
    remove_extra_whitespaces=False,
)


def block_lines(data, offset, start, end):
    """Lines of data (which begins at byte offset) whose first byte is in [start, end)."""
    stream = io.BytesIO(data)
    position = offset
    if start:
        position += len(stream.readline())
    while position < end:
        line = stream.readline()
        if not line:
            raise EOFError(f"Data ended at byte {position}, before {end}")
        yield line
        position += len(line)


def granary_texts(blocks, revision):
    """Map (block index, utt_id) to the transcripts with that ID, in file order.

    A few utt_ids repeat inside a block with different transcripts; the training
    records therefore also give the occurrence.
    """
    texts = {}
    for index, (path, start, end) in enumerate(blocks):
        url = f"https://huggingface.co/datasets/{GRANARY}/resolve/{revision}/{path}"
        offset = max(0, start - 1)
        response = requests.get(
            url,
            headers={
                "Range": f"bytes={offset}-{end + 65535}",
                "Accept-Encoding": "identity",
            },
            timeout=(30, 180),
        )
        response.raise_for_status()
        for line in block_lines(response.content, offset, start, end):
            record = json.loads(line)
            if isinstance(record.get("text"), str):
                texts.setdefault((index, record["utt_id"]), []).append(record["text"])
    return texts


def text_lines(text, maximum_bytes=4000):
    """Split at whitespace so no line exceeds SentencePiece's input-line limit."""
    for line in text.splitlines():
        chunk = ""
        for word in re.findall(r"\S+\s*|\s+", line):
            if chunk and len((chunk + word).encode()) > maximum_bytes:
                yield chunk
                chunk = ""
            chunk += word
        if chunk:
            yield chunk


def training_text(records):
    granary = granary_texts(records["blocks"], records["granary_revision"])
    tedlium = {row["id"]: clean_tedlium(row["text"]) for row in json_records(TEDLIUM)}
    texts = [granary[block, utt][n] for block, utt, n in records["granary"]]
    texts += [tedlium[utt] for utt in records["tedlium"]]
    return "".join(line + "\n" for text in texts for line in text_lines(text))


def main():
    import sentencepiece as spm

    argparse.ArgumentParser(description=__doc__).parse_args()
    records = json.loads(RECORDS.read_text())
    text = training_text(records)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != records["training_text_sha256"]:
        raise ValueError(f"Training text differs from the recorded one: {digest}")
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "training.txt").write_bytes(text.encode())
    # Relative paths keep the model bytes independent of the checkout location.
    with contextlib.chdir(BUILD):
        spm.SentencePieceTrainer.train(
            input="training.txt", model_prefix="tokenizer", minloglevel=2, **TRAINER
        )
    model = (BUILD / "tokenizer.model").read_bytes()
    unchanged = MODEL.exists() and MODEL.read_bytes() == model
    MODEL.write_bytes(model)
    print(
        f"{MODEL.relative_to(ROOT)}: sha256 {hashlib.sha256(model).hexdigest()}"
        + ("; unchanged" if unchanged else "; replaced")
    )


if __name__ == "__main__":
    main()
