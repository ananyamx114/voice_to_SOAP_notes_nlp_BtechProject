# Prepared BeTraC-2000

`prepared.zip` contains 1,600 train, 200 validation and 200 internal-test
encounters selected from official BeTraC training data, with seed 42.
These are synthetic conversations and references, not clinically validated data.
The archive contains only files beneath `data/processed/betrac_2000/`, including
split hashes, selection/validation reports and source attribution. It contains
no audio, raw training archives, model weights or environment files.

After pulling the matching repository code, extract from the repository root:

```powershell
.\.venv\Scripts\python.exe -m zipfile -e datasets/betrac_2000/prepared.zip .
```

Use this on a fresh clone; preserve existing prepared files if you have local
changes. Verify the archive using `prepared.zip.sha256` and each split using
the embedded `split_manifest.json` before training.

The standalone facts cap was removed with user approval; complete DIRECT and
HYBRID training sequences still fit 4,096 tokens without truncation. Transcript
and SOAP limits remain 2,200 and 700 tokens. See
`docs/BETRAC_2000_MIGRATION.md` for methodology and limitations.

Source: BeTraC/betrac-2026, revision
96f7b94e662dac0d1394cfdf55e8b8ae81422663. Distributed under CC BY 4.0;
preserve SOURCE_DATASET_CARD.md and its Labrak et al. (Interspeech 2026)
citation. Our changes are deterministic selection, splitting, speaker-label
conversion and the repository's existing heuristic SOAP mapping. Original text
is retained in the records. No training result is implied.
