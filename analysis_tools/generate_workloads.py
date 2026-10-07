"""Generate workload lengths from public dataset metadata and a SentencePiece model."""

import argparse
from array import array
import csv
import gzip
import io
import hashlib
import importlib.util
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import tomllib
from types import SimpleNamespace

import requests

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(batch=128, hidden=640, vocab=1024, dropout_p=0.1, seed=20261011)
BANDS = (("short", 0.0, 10.0), ("medium", 10.0, 40.16), ("long", 40.16, None))
GRANARY = "nvidia/Granary"
GRANARY_REVISION = "0fe23a860e3570b111e79a05ebeefc53822161e2"
TEDLIUM = (
    "https://media.githubusercontent.com/media/revdotcom/speech-datasets/"
    "c05ab6fd8b4b627d123c922a22a39e993dd37635/"
    "longform_reconstitution/tedlium/tedlium_train_lf_linked.jsonl.gz"
)
VOX_TIMINGS = (
    "https://dl.fbaipublicfiles.com/voxpopuli/annotations/unlabelled_v2.tsv.gz"
)


def json_records(url):
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        lines = (
            gzip.GzipFile(fileobj=response.raw)
            if url.endswith(".gz")
            else response.iter_lines(chunk_size=1024 * 1024)
        )
        for line in lines:
            if line.strip():
                yield json.loads(line)


def clean_tedlium(text):
    """Remove TED-LIUM's <unk> annotations; keep case and punctuation."""
    return " ".join(text.replace("<unk>", " ").split())


def vox_durations():
    """Granary's VoxPopuli duration field is null; use source segment boundaries."""
    print("Reading public VoxPopuli timing annotations...", flush=True)
    durations = {}
    with requests.get(VOX_TIMINGS, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with gzip.GzipFile(fileobj=response.raw) as compressed:
            for row in csv.DictReader(io.TextIOWrapper(compressed), delimiter="\t"):
                if row["event_id"].endswith("_en"):
                    key = row["event_id"] + "_" + row["segment_no"]
                    durations[key] = round(float(row["end"]) - float(row["start"]), 6)
    return durations


def source_records(
    source,
    *,
    config=None,
    split="train",
    revision=None,
    text_column="text",
    duration_column="duration",
):
    """Stream transcript/duration records without decoding audio."""
    if source.lower() == GRANARY.lower():
        from huggingface_hub import HfApi, hf_hub_url

        if config is not None or split != "train":
            raise ValueError(
                "Granary uses all four English ASR manifests; omit --config and use --split train"
            )
        revision = revision or GRANARY_REVISION
        files = sorted(
            f
            for f in HfApi().list_repo_files(
                source, repo_type="dataset", revision=revision
            )
            if f.startswith("en/") and f.endswith("_asr.jsonl")
        )
        if not files:
            raise ValueError("No English ASR manifests found in Granary")
        for file in files:
            print(f"Reading {source}@{revision}/{file}", flush=True)
            timings = vox_durations() if "/voxpopuli/" in file else None
            url = hf_hub_url(source, file, repo_type="dataset", revision=revision)
            for row in json_records(url):
                if row.get("source_lang") != "en" or row.get("taskname") != "asr":
                    raise ValueError(f"Unexpected language/task in {file}")
                duration = row.get("duration")
                if timings is not None and (
                    duration is None or not math.isfinite(duration) or duration <= 0
                ):
                    duration = timings[row["utt_id"]]
                yield dict(text=row.get("text"), duration=duration)
    elif source.startswith("https://"):
        for row in json_records(source):
            text = row[text_column]
            if source == TEDLIUM and isinstance(text, str):
                text = clean_tedlium(text)
            yield dict(text=text, duration=row[duration_column])
    else:
        from datasets import load_dataset

        dataset = load_dataset(
            source, name=config, split=split, revision=revision, streaming=True
        )
        # Select before iteration so audio features are never decoded.
        for row in dataset.select_columns([text_column, duration_column]):
            yield dict(text=row[text_column], duration=row[duration_column])


def corpus_lengths(records, tokenizer):
    """Tokenize transcripts and return paired durations and target lengths."""
    duration, target = array("d"), array("i")
    texts, seconds = [], []
    excluded = 0

    def flush():
        nonlocal excluded
        for d, ids in zip(
            seconds, tokenizer.encode(texts, out_type=int, num_threads=8)
        ):
            if ids:
                duration.append(d)
                target.append(len(ids))
            else:
                excluded += 1
        texts.clear()
        seconds.clear()

    for count, row in enumerate(records, 1):
        text, d = row["text"], row["duration"]
        if not isinstance(text, str) or d is None or not math.isfinite(d) or d <= 0:
            excluded += 1
            continue
        texts.append(text)
        seconds.append(float(d))
        if len(texts) == 4096:
            flush()
        if count % 1_000_000 == 0:
            print(f"  {count:,} records read", flush=True)
    if texts:
        flush()
    print(
        f"{len(duration):,} usable utterances; {excluded:,} invalid/empty records excluded",
        flush=True,
    )
    if not duration:
        raise ValueError(
            "No positive-duration utterances with nonempty token sequences"
        )
    return np.asarray(duration), np.asarray(target)


@lru_cache
def nemo_bucket_estimator():
    """Load NeMo's estimate_duration_bins_2d.py at the benchmark's NeMo commit."""
    project = tomllib.loads((ROOT / "benchmarks/pyproject.toml").read_text())
    source = project["tool"]["uv"]["sources"]["nemo-toolkit"]
    revision = source["rev"]
    path = ROOT / "build/nemo" / revision / "estimate_duration_bins_2d.py"
    if not path.exists():
        url = source["git"].replace(
            "https://github.com/", "https://raw.githubusercontent.com/"
        )
        url += f"/{revision}/scripts/speech_recognition/estimate_duration_bins_2d.py"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    expected = "bad7f688f99abf91fae72287396e02bde012590db55db7093c4bf26d032cb292"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"NeMo bucket estimator checksum mismatch: {path}")
    spec = importlib.util.spec_from_file_location("nemo_duration_bins", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.estimate_duration_buckets


def bucket_batches(duration, target):
    """Use NeMo's 30×2 estimator and strict 2D assignment, then form B128 batches."""
    from nemo.collections.common.data.lhotse.sampling import (
        FixedBucketBatchSizeConstraint2D,
    )

    # The estimator only reads duration and len(input_ids), not audio or token identities.
    cuts = (
        SimpleNamespace(duration=d, input_ids=range(int(u)))
        for d, u in zip(duration, target)
    )
    bins, max_tps = nemo_bucket_estimator()(
        cuts,
        num_buckets=30,
        num_subbuckets=2,
        max_tps=None,
        max_duration=float("inf"),
        token_outlier_threshold=4.0,
        quiet=True,
    )
    constraint = FixedBucketBatchSizeConstraint2D(
        bins, [SETTINGS["batch"]] * len(bins), strict_2d=True, max_ratio=max_tps
    )
    bucket = np.fromiter(
        (
            index
            if (index := constraint.select_bucket(bins, example_len=(d, u))) is not None
            else -1
            for d, u in zip(duration, target)
        ),
        dtype=np.int32,
        count=len(duration),
    )
    accepted = np.flatnonzero(bucket >= 0)
    grouped = accepted[np.argsort(bucket[accepted], kind="stable")]
    counts = np.bincount(bucket[accepted], minlength=len(bins))
    offsets = np.r_[0, counts.cumsum()]
    batch = SETTINGS["batch"]
    members = np.empty((int((counts // batch).sum()), batch), dtype=np.int64)
    rng = np.random.default_rng(SETTINGS["seed"])
    cursor = 0
    for a, b in zip(offsets[:-1], offsets[1:]):
        indices = rng.permutation(grouped[a:b])
        full = len(indices) // batch
        members[cursor : cursor + full] = indices[: full * batch].reshape(full, batch)
        cursor += full
    print(
        f"{len(members):,} complete batches; {len(duration) - len(accepted):,} outside bucket/TPS limits; "
        f"{len(accepted) - members.size:,} utterances in incomplete batches",
        flush=True,
    )
    return members


def encoder_lengths(duration):
    return np.ceil(duration / 0.08 - 1e-9).astype(np.int64)


def centrality(values):
    """Equal-weight distance to feature medians, scaled by interquartile range."""
    values = np.array(values, dtype=float, copy=True)
    values[:, :2] = np.log(values[:, :2])
    quartiles = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
    spread = quartiles[2] - quartiles[0]
    active = spread > 0
    return (np.abs(values[:, active] - quartiles[1, active]) / spread[active]).sum(
        axis=1
    )


def select_workloads(duration, target, bands=BANDS):
    members = bucket_batches(duration, target)
    features = np.empty((len(members), 3))
    # Bound temporary matrices independently of corpus size.
    for start in range(0, len(members), 8192):
        indices = members[start : start + 8192]
        d, u = duration[indices], target[indices].astype(np.int64)
        t = encoder_lengths(d)
        rows = (t * (u + 1)).sum(axis=1)
        padded = SETTINGS["batch"] * t.max(axis=1) * (u.max(axis=1) + 1)
        features[start : start + len(indices)] = np.column_stack(
            (d.mean(axis=1), rows, 1 - rows / padded)
        )
    workloads = {}
    for name, lower, upper in bands:
        eligible = np.flatnonzero(
            (features[:, 0] > lower)
            & (True if upper is None else features[:, 0] <= upper)
        )
        if not len(eligible):
            raise ValueError(f"No complete batches in the {name} duration band")
        # argmin preserves source batch order when distances tie.
        index = int(eligible[np.argmin(centrality(features[eligible]))])
        chosen = members[index]
        encoder = encoder_lengths(duration[chosen])
        workloads[name] = dict(
            mean_duration_s=float(duration[chosen].mean()),
            lengths=np.column_stack((encoder, target[chosen])).tolist(),
        )
        print(
            f"{name}: batch {index}, {features[index, 0]:.2f} s, {int(features[index, 1]):,} valid positions"
        )
    return workloads


def generate(
    dataset=GRANARY,
    *,
    tokenizer=ROOT / "data/tokenizer/tokenizer.model",
    config=None,
    split="train",
    revision=None,
    long_dataset=None,
    long_config=None,
    long_revision=None,
    text_column="text",
    duration_column="duration",
):
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    if processor.vocab_size() + 1 != SETTINGS["vocab"]:
        raise ValueError("Tokenizer must have 1023 entries plus the RNN-T blank")
    if long_dataset is None and dataset.lower() == GRANARY.lower():
        long_dataset = TEDLIUM
    sources = [(dataset, config, revision, BANDS[:2] if long_dataset else BANDS)]
    if long_dataset:
        sources.append((long_dataset, long_config, long_revision, BANDS[2:]))
    workloads = {}
    for source, name, commit, bands in sources:
        records = source_records(
            source,
            config=name,
            split=split,
            revision=commit,
            text_column=text_column,
            duration_column=duration_column,
        )
        duration, target = corpus_lengths(records, processor)
        workloads.update(select_workloads(duration, target, bands))
    return dict(SETTINGS, workloads=workloads)


def write_workloads(path, workloads):
    # Keep each [T, U] pair on one line.
    text = json.dumps(workloads, indent=2, allow_nan=False)
    text = re.sub(r"\[\s+(\d+),\s+(\d+)\s+\]", r"[\1, \2]", text)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", nargs="?", default=GRANARY, help="Hugging Face dataset repository"
    )
    parser.add_argument(
        "--config",
        help="HF dataset configuration; Granary uses all English ASR sources",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--revision", help="HF commit; defaults to the paper revision for Granary"
    )
    parser.add_argument(
        "--long-dataset",
        help="Separate HF repo or JSONL URL for long; Granary defaults to Rev TED-LIUM",
    )
    parser.add_argument("--long-config")
    parser.add_argument("--long-revision")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--duration-column", default="duration", help="Duration in seconds"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "data/tokenizer/tokenizer.model"
    )
    parser.add_argument("--output", type=Path, default=Path("workloads.json"))
    args = parser.parse_args()
    output = vars(args).pop("output")
    write_workloads(output, generate(**vars(args)))
    print(output)


if __name__ == "__main__":
    main()
