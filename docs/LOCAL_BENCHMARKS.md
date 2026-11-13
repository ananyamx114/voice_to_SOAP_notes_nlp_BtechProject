# Local audio benchmark setup

Checked on 13 November 2026. Hardware: MacBook Air M1, 8 GB unified RAM;
approximately 35 GiB available before these downloads. No NVIDIA GPU is needed
for the CPU smoke experiment. More elapsed time cannot guarantee that every
model/context fits in 8 GB. Run stages sequentially, measure memory and latency,
and use appropriately sized/quantized LLMs. Large-model training is a separate,
unproven workload on this laptop.

## Downloaded and checked

All selected files were downloaded at immutable Hugging Face revisions. Large
files were checked against upstream SHA-256 and size; every file has a recorded
local SHA-256 in its ignored `download_manifest.json`. Dataset cards/attribution
are retained beside the downloads. No dataset or model credentials were needed.

| Dataset | Local files | Bytes | Intended use |
| --- | --- | ---: | --- |
| [BeTraC](https://huggingface.co/datasets/BeTraC/betrac-2026) | `data/raw/betrac/`: card and full dev archive | 469,427,073 | Synthetic audio/transcript/SOAP comparisons; 400 available encounters |
| [Eka](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset) | `data/raw/eka/`: card, dataset info, eight English and one Hindi Parquet shard | 281,398,423 | ASR test benchmark; actual Parquet row counts: 3,619 en + 320 hi |
| [Sully](https://huggingface.co/datasets/SullyHQ/open-clinical-conversations) | `data/raw/sully-smoke/`: one PriMock57 MP3, encounter/audio metadata, card and attribution | 16,539,969 | Role-played human consultation sample; audio decoded at 16 kHz mono, 457.92 seconds |

Pinned revisions:

- BeTraC: `96f7b94e662dac0d1394cfdf55e8b8ae81422663`.
- Eka: `433e66037c8aebb8f396b938624a5ab6afc4ca36`.
- Sully: `06053cd9be2ea5fbfaefebb1db5be907b53a8c44`.

BeTraC's card calls the archive `dev`, while its metadata calls the split
`validation`; both labels are preserved by the importer. The archive contains
1,600 members: 400 sets of Opus, transcript, SOAP text and metadata. Two real
samples were exported and decoded locally. They last 343.65 and 240.85 seconds.
These notes are synthetic targets without claim-level evidence annotations.

Eka's actual shard sizes total about 268 MiB. Its README's historical download
size fields differ; use the revision/file manifest and observed row counts for
this run. Its short clips include narration and conversation and have no SOAP
targets. The four-clip smoke selection is the first two en and first two hi rows,
not a representative or randomized research evaluation set.

Sully's card identifies 536 encounters and 329 with audio. The downloaded
metadata has those counts, but only **one** audio encounter was downloaded.
Its sources include the same 207 ACI-Bench encounters already in this project:
do not treat Sully as wholly independent or mix those records across splits.
PriMock57 reference notes are clinical notes, not automatically gold cited SOAP.

## Deferred sources and access

| Source | Actual access/compatibility finding |
| --- | --- |
| [DISPLACE-M](https://huggingface.co/datasets/Displace-M/Displace2026) | Card says use is limited to challenge participants under signed terms, despite also containing CC BY text. Await registration/permission and organizer instructions before downloading/using it. Not a Marathi benchmark. |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) | Gated: accept the conditions in your own HF account. Select bounded Hindi/Marathi evaluation subsets; do not download the entire corpus to this laptop. General speech is not clinical SOAP data. |
| [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | Gated: accept the conditions. Indian-accented English ASR, not SOAP or Marathi ASR. |

## Reproduce downloads and preparation

```bash
.venv/bin/python scripts/download_benchmarks.py --dataset all
.venv/bin/python -m voice_to_soap.benchmark_datasets \
  --archive data/raw/betrac/data/dev-00000.tar \
  --output-dir data/processed/betrac-smoke --limit 2
.venv/bin/python -m voice_to_soap.cli import-eka-asr \
  --dataset-dir data/raw/eka --outdir data/processed/eka-smoke \
  --manifest-out data/manifests/eka-smoke.csv --limit-per-language 2
```

The downloader skips intact large files, pins each run's upstream revision, checks
hashes and preserves at least 12 GiB free. It selects dev/test data and a Sully
sample; no BeTraC training shards are downloaded. Output preparation refuses to
overwrite an existing directory/manifest. Use a new output name for repeat runs.

BeTraC outputs `audio_manifest.csv` (empty input transcript), `gold_manifest.csv`
(explicit doctor/patient labels), original Opus/transcript/SOAP/metadata and a
preparation report. Reference transcript and note fields are for scoring. The
pipeline reads only `transcript` as text input; the audio manifest keeps it empty.
Do not relabel development data as an untouched test set after tuning on it.

## Real local ASR experiment

```bash
.venv/bin/python -m pip install -r requirements-local-asr.txt
HF_HOME=data/models/huggingface .venv/bin/python -m voice_to_soap.cli evaluate-asr \
  --manifest data/manifests/eka-smoke.csv \
  --model openai/whisper-small --device cpu --limit-per-language 2 \
  --outdir data/outputs/eka-whisper-small-local-smoke
```

Whisper Small is the multilingual model, not the English-only `.en` checkpoint.
Weights are under ignored `data/models/huggingface`. Its cached revision for this
run is `973afd24965f72e36ca33b3055d56a652f456b4d`. CPU inference uses the existing
word-timestamp pipeline. The installed Transformers warns that chunked seq2seq
ASR is experimental; evaluate a validated long-form path before benchmarking
full consultations. The four clips here are each under 12 seconds.

The command writes `predictions.jsonl` and `metrics.json` only on completion.
Those files, rather than mock tests or dataset publishers' scores, are the
evidence for actual local inference. Clinical factuality, citation correctness,
speaker DER, SOAP quality and translation adequacy are not measured by WER/CER.

The actual run completed: English WER 0.20 and Hindi WER 3.3261 on two clips
each. Hindi output had substantial excess text; this is a failure to investigate,
not acceptable clinical performance. WER can exceed 1 because of insertions.
See [the saved aggregate results](../results/local-benchmark-setup/README.md).
No SOAP or diarization inference was run. All 3,939 Eka clips were exported via
the existing importer, and all 3,939 manifest audio paths exist. Unit tests:
75 passed. Installed local ASR versions are recorded separately in
`requirements-local-asr-macos-py312.lock.txt`; the initial lightweight lock is
unchanged. Latency and peak memory require instrumentation
for a substantive benchmark.

## Connections needed for later phases

Public downloads work now. For gated sources, sign into your own Hugging Face
account, accept each repository's conditions, and configure a read-scoped token
locally as `HF_TOKEN` when needed. Do not paste tokens into chat or commit them.
pyannote additionally requires access to its selected diarization checkpoint.
DISPLACE-M requires the separate participation/permission process.

Local SOAP generation still needs an inference runtime/model. On this machine,
start with a quantized small checkpoint and limited context, then measure it;
do not claim the default 7B pipeline or every proposed model is ready. Ollama
`qwen2.5:3b` is a 1.9 GB Q4_K_M candidate with its own research license, not an
Apache-licensed substitute. No LLM runtime or SOAP inference has been installed
or validated in this dataset preparation step.

## Team collaboration

The configured origin is `ananyamx114/voice_to_SOAP_notes_nlp_BtechProject`.
The team has chosen to work on `main`, using individual GitHub accounts and
separate local clones. Coordinate file ownership, make small commits, fetch and
review remote changes before integrating, and never force-push shared main.
Preserve local edits and resolve conflicts before pushing. Keep task status,
commands and design decisions in repository docs/PR descriptions, so another
Codex session can inspect them. Conversations are not automatically shared.
See `TEAM_SETUP.md` for the current handoff and publication workflow.

Use separate OpenAI accounts: [account sharing is not supported](https://help.openai.com/en/articles/10471989-openai-account-sharing-policy).
[ChatGPT Plus](https://help.openai.com/en/articles/6950777-what-is-chatgpt-plus)
does not include ordinary API billing. Codex authentication and the project's
local model inference are separate; no ChatGPT credential becomes a local model
weight or a general application API key.
