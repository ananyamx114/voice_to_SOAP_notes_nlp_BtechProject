"""Pinned, full-checkpoint Whisper sequential decoding with explicit provenance."""
from functools import lru_cache, wraps
from importlib.metadata import version
import math
import re

import numpy as np


GENERATION_KEYS = {"max_new_tokens", "num_beams", "condition_on_prev_tokens",
                   "compression_ratio_threshold", "temperature", "logprob_threshold",
                   "no_speech_threshold"}


def omit_unused_encoder_attention_outputs(model):
    """Word alignment needs decoder cross-attention, not encoder attention outputs.

    Transformers generation requests output_attentions for both stacks. Keep
    encoder computations unchanged, but do not retain its per-layer matrices.
    This does not prune layers/weights or alter decoder attention collection.
    """
    encoder = model.get_encoder()
    original = encoder.forward

    @wraps(original)
    def forward(*args, **kwargs):
        kwargs["output_attentions"] = False
        return original(*args, **kwargs)

    encoder.forward = forward


@lru_cache(maxsize=1)
def load_sequential_pipeline(model_id: str, revision: str, device: str, precision: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    dtype = {"float16": torch.float16, "float32": torch.float32}[precision]
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True,
        use_safetensors=True, attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    omit_unused_encoder_attention_outputs(model)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    return pipeline("automatic-speech-recognition", model=model,
                    tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                    device=device, torch_dtype=dtype, batch_size=1)


def sequential_transcribe(audio, sample_rate: int, model_id: str, device: str,
                         language: str | None, options: dict) -> dict:
    from .audio import _resolve_asr_device

    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("ASR needs a nonempty finite mono waveform")
    if sample_rate != 16000:
        raise ValueError("Sequential Whisper input must be resampled to 16000 Hz")
    revision = options.get("revision", "")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Sequential ASR requires an immutable 40-character model revision")
    selected_device = _resolve_asr_device(device)
    precision = options.get("precision", "auto")
    if precision == "auto":
        precision = "float16" if selected_device.split(":")[0] in {"cuda", "mps"} else "float32"
    if precision not in {"float16", "float32"}:
        raise ValueError("ASR precision must be auto, float16 or float32")
    timestamps = options.get("timestamps", "word")
    if timestamps not in {"word", "segment"}:
        raise ValueError("ASR timestamps must be word or segment")
    generation = dict(options.get("generation", {}))
    unknown = generation.keys() - GENERATION_KEYS
    if unknown:
        raise ValueError(f"Unsupported ASR decoding settings: {sorted(unknown)}")
    if "temperature" in generation and isinstance(generation["temperature"], list):
        generation["temperature"] = tuple(generation["temperature"])
    generation["task"] = "transcribe"
    if language:
        generation["language"] = language
    asr = load_sequential_pipeline(model_id, revision, selected_device, precision)
    last_progress = -1

    def monitor_progress(batch):
        nonlocal last_progress
        seek, total = batch.detach().cpu().tolist()[0]
        percent = int(100 * seek / total) if total else 0
        if percent // 5 != last_progress:
            print(f"  sequential decoding: {percent}% of source frames reached", flush=True)
            last_progress = percent // 5

    inference_generation = {**generation, "monitor_progress": monitor_progress}
    # No chunk_length_s: keep Whisper's native sequential long-form algorithm.
    result = asr({"raw": audio, "sampling_rate": sample_rate},
                 return_timestamps="word" if timestamps == "word" else True,
                 generate_kwargs=inference_generation)
    chunks, warnings = [], []
    duration = len(audio) / sample_rate
    for index, item in enumerate(result.get("chunks") or []):
        text = item.get("text", "").strip()
        if not text:
            continue
        start, end = item.get("timestamp") or (None, None)
        for value in (start, end):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("ASR returned a nonfinite timestamp")
        start = float(start) if start is not None else None
        end = float(end) if end is not None else None
        if start is None or end is None:
            warnings.append({"chunk": index, "issue": "missing_timestamp"})
        elif start < 0 or end < start or end > duration + 0.1:
            warnings.append({"chunk": index, "issue": "timestamp_outside_source_or_reversed"})
        chunks.append({"text": text, "start": start, "end": end,
                       "confidence": item.get("confidence")})
    text = result.get("text", "").strip()
    if not text:
        raise RuntimeError("Whisper returned no transcript")
    if not chunks:
        chunks = [{"text": text, "start": None, "end": None, "confidence": None}]
        warnings.append({"issue": "no_timestamp_chunks"})
    return {"text": text, "chunks": chunks, "warnings": warnings,
            "metadata": {"model_id": model_id, "revision": revision, "device": selected_device,
                         "precision": precision, "quantization": "none",
                         "algorithm": "sequential", "timestamps": timestamps,
                         "encoder_attention_outputs": False,
                         "generation": generation, "torch_version": version("torch"),
                         "transformers_version": version("transformers")}}


def speech_context_regions(segments: list[dict], duration: float, join_gap: float = 2.0) -> list[dict]:
    """Keep interior pauses/context; skip only gaps larger than the explicit threshold."""
    if not math.isfinite(join_gap) or join_gap < 0 or duration <= 0:
        raise ValueError("Invalid speech-context duration or gap")
    regions = []
    previous = 0.0
    for item in segments:
        start = float(item["start_seconds"])
        end = float(item["end_seconds"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration:
            raise ValueError("VAD interval outside source")
        if start < previous:
            raise ValueError("VAD intervals overlap or are unsorted")
        if regions and start - regions[-1]["end_seconds"] <= join_gap:
            regions[-1]["end_seconds"] = end
        else:
            regions.append({"start_seconds": start, "end_seconds": end})
        previous = end
    if not regions:
        raise ValueError("No VAD speech regions available for ASR")
    return regions
