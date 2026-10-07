# Workloads and tokenizer

`workloads/workloads.json` contains shared benchmark settings and three B128
batches. Each `lengths` entry is `[T, U]`: encoder length and target length for
one utterance, in execution order. `mean_duration_s` supplies the figure labels.
Row counts and padding are calculated from these pairs, not stored separately.

## Selection and regeneration

The short and medium batches come from the English subset of Granary; the long
batch comes from the linked long-form training split of TED-LIUM 3. The
supplied batches were selected with an earlier, approximate 30×2 equal-mass
bucketer.

Workload generation uses NeMo's strict 2D bucket assignment and its
`estimate_duration_bins_2d.py`, fetched at the benchmark's NeMo commit.
It requests 30 duration buckets and two token sub-buckets, with NeMo's
four-standard-deviation token-rate filtering and duplicate-bin handling.
The estimated token-rate limits also apply during assignment. Utterances are
shuffled within each bucket with seed 20261011 and grouped into complete B128
batches. Out-of-bucket and incomplete-batch exclusions are reported separately.

Regeneration therefore selects different batches from the supplied ones; the
supplied lengths remain the inputs of the existing measurements.

Using batch mean duration, the bands are short `(0, 10]` seconds, medium
`(10, 40.16]`, and long `>40.16`. Within each band, selection minimizes the sum
of distances to the medians of log mean duration, log valid alignment positions
and padding fraction, scaled by each feature's interquartile range. Constant
features are ignored; ties use source batch order. No benchmark measurements
enter selection. These are three selected batches, not corpus-wide averages.

Generate directly from public sources, with no prepared cache or audio download:

```bash
uv run --locked --project benchmarks python analysis_tools/generate_workloads.py nvidia/Granary --output workloads.json
```

The Granary preset reads all English ASR manifests at revision
`0fe23a860e3570b111e79a05ebeefc53822161e2`, tokenizes with the supplied model,
builds the 2D buckets and selects short/medium batches. Missing VoxPopuli
durations are recovered from its public segment timing annotations. For long,
it reads [Rev's linked TED-LIUM training manifest](https://github.com/revdotcom/speech-datasets/tree/c05ab6fd8b4b627d123c922a22a39e993dd37635/longform_reconstitution/tedlium).
This is not the standard short-segment TED-LIUM split on Hugging Face. Literal
`<unk>` annotations are removed from TED-LIUM text; case and punctuation are kept.

To replace the published file, run `make workloads`. This processes the full
English corpus, not a sample, so it still requires substantial metadata transfer
and tokenization time. The output is the workload JSON; the NeMo script is
cached under `build/nemo/`. Invalid durations and empty token sequences are
excluded, with counts printed alongside incomplete-batch counts.

Other HF repositories can be used with the same selection rule:

```bash
uv run --locked --project benchmarks python analysis_tools/generate_workloads.py org/dataset --config en \
  --revision COMMIT --text-column text --duration-column duration \
  --output workloads.json
```

The dataset must provide transcripts and durations in seconds. `--split`
defaults to `train`; `--config` is optional for single-config datasets.
All three bands use that dataset unless `--long-dataset` supplies another HF
repository or a public JSONL URL. HF sources may need authentication under
their own access rules. Audio is not decoded to infer missing durations.
If a band has no complete batch, generation fails.

Encoder lengths are `ceil(duration_s / 0.08)`. Valid positions are
`sum(T * (U + 1))`; padded positions are `128 * max(T) * (max(U) + 1)`.
Selection uses source row order and the fixed random seed. Changing the source,
revision or tokenizer can change the selected batches; it does not regenerate
measurements for them. The original workload definitions are also recorded in
the published comparison and ablation manifests under `results/raw/`.

## Tokenizer

`tokenizer/tokenizer.model` is the SentencePiece BPE model used to derive
target lengths: 1,023 entries, plus the RNN-T blank for a 1,024-entry joint
vocabulary. `make build-tokenizer` regenerates it byte for byte from public
sources. It fetches the 20,480 transcripts listed in
`tokenizer/training_records.json`: 16,384 Granary records from 256 byte blocks
of the English ASR manifests at the pinned revision, and 4,096 records of the
linked TED-LIUM 3 training manifest. It checks the assembled training text
against its recorded SHA-256, trains with the arguments NeMo's
`process_asr_text_tokenizer.py` passes to SentencePiece (`--no_lower_case`,
`--spe_max_sentencepiece_length 512`), and writes the model. The download is
about 130 MB. `provenance.json` describes the sources and settings. The
tokenizer is not covered by the repository's Apache-2.0 license.

The benchmark reads only these lengths; no audio or transcripts are
distributed. Dataset citations are in the paper's bibliography.
