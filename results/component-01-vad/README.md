# Actual Silero VAD component execution

Executed 13 November 2026 after publishing implementation commit `6c88124`.
Uses Silero VAD 6.2.3's released TorchScript model on CPU via the exact same
detect_speech function used by the integrated pipeline. No mock model was used.
See [commands and validation](../../docs/COMPONENT_01_VAD.md).

| Complete encounter | Source seconds | Detected speech seconds | Regions | Processing seconds |
| --- | ---: | ---: | ---: | ---: |
| dialog_0060_0120 | 343.646 | 315.000 | 97 | 2.727 |
| dialog_0060_0121 | 240.850 | 212.900 | 83 | 1.141 |

All 180 exported region WAVs decoded successfully at mono 16 kHz. Their durations
match their original-recording timestamp intervals. Interval bounds/order and
total speech-duration consistency passed. `summary.json` records exact package
versions, model size/hash, parameters, timings and checks without absolute paths
or consultation/audio content. Local artifacts stay under ignored data/outputs.

Timings include per-encounter loading, model initialization, detection and WAV
writing, but exclude process startup and metadata writing. No peak-memory or
cross-machine performance claim is made. PyTorch emitted a TorchScript loading
deprecation warning; both encounters completed.

These are two synthetic BeTraC dev encounters, not a representative clinical
VAD benchmark. Manual boundary listening was not performed; speech recall and
short-word/negation preservation remain unmeasured. A region count or high speech
fraction does not prove accuracy. Inspect original audio around segment edges
before accepting settings for subsequent ASR or running larger evaluation.
