---
dataset_info:
  - config_name: default
    features:
      - name: opus
        dtype: binary
      - name: transcript.txt
        dtype: string
      - name: soap.txt
        dtype: string
      - name: json
        dtype: string
    splits:
      - name: validation
        num_examples: 400
        num_bytes: 469422080
      - name: train
        num_examples: 7200
        num_bytes: 8661381120
license: cc-by-4.0
task_categories:
  - automatic-speech-recognition
  - summarization
language:
  - en
tags:
  - medical
  - doctor-patient
  - webdataset
  - soap-notes
  - betrac
size_categories:
  - 1K<n<10K
pretty_name: BeTraC 2026 - DoPaCo Audio Dataset
---

# BeTraC 2026 - Synth-DoPaCo Audio Dataset

Synthetic doctor-patient conversations with audio, transcripts, dialog metadata, and SOAP note summaries.

## Limitations and intended use

All conversations in this dataset are synthetic. No real patients, clinicians or clinical records were involved. The dialogues were written by a language model from sampled personas, the audio was produced by speech synthesis, and the reference notes were written by a language model.

The data is intended for research on long-form audio understanding and note generation. It is not clinical data and it is not a substitute for real clinical recordings. The conversations have not been validated for medical accuracy, and nothing in them should be used as medical advice or to train or validate a system for clinical use without evaluation on real data.

Known gaps to real visits include: two speakers only, English only, primary care first visits only, no accents, cleaner audio than real exam rooms, longer and more formal doctor turns, and no audible symptoms or prosody that carry clinical meaning. See the paper for details.

## Dataset Splits

| Split | Dialogs | Shards | Size |
|---|---|---|---|
| `dev` | 400 | 1 | 469 MB |
| `train` | 7,200 | 9 | 8.7 GB |


## File Format

This dataset uses the [WebDataset](https://github.com/webdataset/webdataset) format (tar archives).
Each sample contains 4 files sharing the same key (e.g., `dialog_0060_0120`):

| Extension | Content |
|---|---|
| `.opus` | Opus-compressed audio (16 kHz mono) |
| `.transcript.txt` | Full transcript of the doctor-patient dialog |
| `.json` | Dialog metadata (personas, generation parameters, dialog turns) |
| `.soap.txt` | Target SOAP note summary |

## Usage

```python
import webdataset as wds
import json

dataset = wds.WebDataset("path/to/dev-00000.tar", shardshuffle=False)

for sample in dataset:
    key = sample["__key__"]
    audio_bytes = sample["opus"]        # raw Opus bytes
    transcript = sample["transcript.txt"].decode("utf-8")
    soap_note = sample["soap.txt"].decode("utf-8")
    metadata = json.loads(sample["json"])
    print(f"{key}: {len(audio_bytes)} bytes audio, {len(transcript)} chars transcript")
    break
```

### Streaming from Hugging Face Hub

```python
import webdataset as wds
from huggingface_hub import get_token

token = get_token()
url = "https://huggingface.co/datasets/BeTraC/betrac-2026/resolve/main/data/train-{00000..00008}.tar"
url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {token}'"

dataset = wds.WebDataset(url, shardshuffle=False)
for sample in dataset:
    print(sample["__key__"])
    break
```

### Decoding Audio

The `.opus` files are Ogg/Opus containers. Decode with `soundfile`, `torchaudio`, or `ffmpeg`:

```python
import soundfile as sf
import io

audio_data, sample_rate = sf.read(io.BytesIO(sample["opus"]))
```

## License

This dataset is released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{labrak26_interspeech,
  title     = {Generating Synthetic Doctor-Patient Conversations for Long-form Audio Summarization},
  author    = {Labrak, Yanis and Gr{\"u}nert, David and Baroudi, S{\'e}verin and Chun, Jiyun and Cyrta, Pawe{\l} and Burdisso, Sergio and Hassoon, Ahmed and Liu, David and Rothschild, Adam and Van Deusen, Reed and Motlicek, Petr and Perrault, Andrew and Marxer, Ricard and Schaaf, Thomas},
  booktitle = {Interspeech 2026},
  year      = {2026},
  pages     = {4384--4389},
  doi       = {10.21437/Interspeech.2026-2901},
  issn      = {2958-1796},
  url       = {https://www.isca-archive.org/interspeech_2026/labrak26_interspeech.html}
}
```

## Acknowledgments

The Synth-DoPaCo dataset used in BeTraC 2026 was created by the Play-Your-Part team during the [JSALT 2025](https://jsalt2025.fit.vut.cz/) workshop, organized by the Center for Language and Speech Processing at Johns Hopkins University and held at Brno University of Technology.
