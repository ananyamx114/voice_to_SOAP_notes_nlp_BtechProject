from pathlib import Path
from typing import Any
import logging
import math
from functools import lru_cache
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


def load_audio(path: str | Path, target_rate: int) -> tuple[np.ndarray, int]:
    if target_rate <= 0:
        raise ValueError("Target sample rate must be positive")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[0] == 0:
        raise ValueError(f"Audio file contains no samples: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio file contains non-finite sample values: {path}")
    mono = audio.mean(axis=1)
    if sample_rate != target_rate:
        from math import gcd
        divisor = gcd(sample_rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, sample_rate // divisor).astype(np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"Audio file contains no samples: {path}")
    return mono, target_rate


def prepare_audio_file(
    source_path: str | Path,
    output_path: str | Path,
    target_rate: int = 16000,
) -> dict[str, Any]:
    """Write a mono WAV at the target rate and return a sidecar metadata record."""
    source = Path(source_path).resolve()
    destination = Path(output_path).resolve()
    if source == destination:
        raise ValueError("Output audio path must differ from the original input path")
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    if target_rate <= 0:
        raise ValueError("Target sample rate must be positive")

    try:
        info = sf.info(str(source))
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Could not read audio metadata from {source}: {exc}") from exc
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError(f"Audio file has invalid dimensions or sample rate: {source}")

    try:
        waveform, sample_rate = load_audio(source, target_rate)
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Could not decode audio file {source}: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), waveform, sample_rate, subtype="PCM_16")
    duration = len(waveform) / sample_rate
    metadata = {
        "source_path": str(source),
        "prepared_path": str(destination),
        "source_sample_rate": int(info.samplerate),
        "prepared_sample_rate": int(sample_rate),
        "source_channels": int(info.channels),
        "prepared_channels": 1,
        "source_duration_seconds": float(info.duration),
        "prepared_duration_seconds": float(duration),
        "sample_count": int(len(waveform)),
        "peak_amplitude": float(np.max(np.abs(waveform))),
        "clipped_sample_fraction": float(np.mean(np.abs(waveform) >= 1.0)),
    }
    if not math.isclose(info.duration, duration, rel_tol=0.01, abs_tol=0.02):
        raise RuntimeError(
            f"Audio duration changed unexpectedly while preparing {source}: "
            f"{info.duration:.3f}s -> {duration:.3f}s"
        )
    return metadata


def normalize_speech_segments(
    raw_segments: list[dict[str, Any]],
    duration_seconds: float,
) -> list[dict[str, float]]:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Audio duration must be a positive finite value")

    segments: list[dict[str, float]] = []
    for index, segment in enumerate(raw_segments):
        try:
            start = float(segment.get("start", segment.get("start_seconds")))
            end = float(segment.get("end", segment.get("end_seconds")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"VAD segment {index} needs numeric start and end times") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"VAD segment {index} has a non-finite timestamp")
        start = min(max(start, 0.0), duration_seconds)
        end = min(max(end, 0.0), duration_seconds)
        if end <= start:
            continue
        segments.append({"start_seconds": start, "end_seconds": end})

    segments.sort(key=lambda segment: segment["start_seconds"])
    merged: list[dict[str, float]] = []
    for segment in segments:
        if merged and segment["start_seconds"] <= merged[-1]["end_seconds"]:
            merged[-1]["end_seconds"] = max(
                merged[-1]["end_seconds"], segment["end_seconds"]
            )
        else:
            merged.append(segment)
    return merged


def detect_speech(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 100,
) -> list[dict[str, float]]:
    if sample_rate not in {8000, 16000}:
        raise ValueError("Silero VAD expects audio sampled at 8000 or 16000 Hz")
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("VAD input must be a non-empty mono waveform")
    if not np.isfinite(audio).all():
        raise ValueError("VAD input contains non-finite sample values")
    if not 0 < threshold < 1:
        raise ValueError("VAD threshold must be between 0 and 1")
    if min_speech_duration_ms < 0 or min_silence_duration_ms < 0 or speech_pad_ms < 0:
        raise ValueError("VAD duration settings cannot be negative")

    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "Silero VAD is not installed; install the audio dependencies from requirements.txt"
        ) from exc

    model = load_silero_vad()
    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    raw_segments = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )
    return normalize_speech_segments(raw_segments, len(audio) / sample_rate)


def transcribe_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[dict[str, float]],
    model_id: str | None,
    device: str = "auto",
    language: str | None = None,
    chunk_length_s: int = 30,
    stride_length_s: int = 5,
) -> list[dict[str, Any]]:
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("ASR input must be a non-empty mono waveform")
    if sample_rate <= 0:
        raise ValueError("ASR sample rate must be positive")

    duration = len(audio) / sample_rate
    valid_segments = normalize_speech_segments(segments, duration)
    if not valid_segments:
        raise RuntimeError("VAD found no speech segments to transcribe")

    words: list[dict[str, Any]] = []
    for segment in valid_segments:
        start_sample = max(0, int(segment["start_seconds"] * sample_rate))
        end_sample = min(len(audio), int(segment["end_seconds"] * sample_rate))
        if end_sample <= start_sample:
            continue
        segment_words = transcribe(
            audio[start_sample:end_sample],
            sample_rate,
            model_id,
            device,
            language=language,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
        )
        for word in segment_words:
            local_start = word["start"]
            local_end = word["end"]
            words.append({
                **word,
                "start": (
                    segment["start_seconds"] + local_start
                    if local_start is not None else segment["start_seconds"]
                ),
                "end": (
                    segment["start_seconds"] + local_end
                    if local_end is not None else segment["end_seconds"]
                ),
                "vad_segment_start": segment["start_seconds"],
                "vad_segment_end": segment["end_seconds"],
            })
    if not words:
        raise RuntimeError("ASR returned no text for any detected speech segment")
    return words


def save_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[dict[str, float]],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Segment source must be a non-empty mono waveform")
    duration = len(audio) / sample_rate
    normalized = normalize_speech_segments(segments, duration)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    written = []
    for index, segment in enumerate(normalized, start=1):
        start_sample = max(0, int(segment["start_seconds"] * sample_rate))
        end_sample = min(len(audio), int(segment["end_seconds"] * sample_rate))
        if end_sample <= start_sample:
            continue
        output_path = output_root / f"speech_{index:04d}.wav"
        sf.write(
            str(output_path),
            audio[start_sample:end_sample],
            sample_rate,
            subtype="PCM_16",
        )
        written.append({
            "index": index,
            "start_seconds": start_sample / sample_rate,
            "end_seconds": end_sample / sample_rate,
            "duration_seconds": (end_sample - start_sample) / sample_rate,
            "audio_path": str(output_path.resolve()),
        })
    return written


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 30,
    padding_ms: int = 240,
    min_speech_dbfs: float = -48,
) -> tuple[np.ndarray, float, float]:
    """Trim leading/trailing quiet using frame energy; retain interior pauses."""
    frame = max(1, int(sample_rate * frame_ms / 1000))
    usable = len(audio) - len(audio) % frame
    if usable == 0:
        raise ValueError("Audio is shorter than one VAD frame")
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    dbfs = 20 * np.log10(np.maximum(rms, 1e-8))
    peak_dbfs = float(dbfs.max())
    threshold = max(min_speech_dbfs, peak_dbfs - 35)
    speech = np.flatnonzero(dbfs >= threshold)
    if len(speech) == 0:
        raise ValueError("VAD found no speech above the configured threshold")
    pad = int(sample_rate * padding_ms / 1000)
    start = max(0, int(speech[0] * frame) - pad)
    end = min(len(audio), int((speech[-1] + 1) * frame) + pad)
    return audio[start:end], start / sample_rate, end / sample_rate


def _resolve_asr_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def _load_asr_pipeline(model_id: str, device: str):
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and Transformers to run ASR") from exc

    device_type = device.split(":", maxsplit=1)[0]
    dtype = torch.float16 if device_type == "cuda" else torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=dtype,
    )


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    model_id: str | None,
    device: str = "auto",
    language: str | None = None,
    chunk_length_s: int = 30,
    stride_length_s: int = 5,
) -> list[dict[str, Any]]:
    if not model_id:
        raise RuntimeError("Audio has no supplied transcript and audio.asr_model_id is unset")
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("ASR input must be a non-empty mono waveform")
    if sample_rate <= 0:
        raise ValueError("ASR sample rate must be positive")
    if chunk_length_s <= 0 or stride_length_s < 0 or stride_length_s * 2 >= chunk_length_s:
        raise ValueError("ASR chunk/stride settings are invalid")
    if not np.isfinite(audio).all():
        raise ValueError("ASR input contains non-finite sample values")

    selected_device = _resolve_asr_device(device)
    asr = _load_asr_pipeline(model_id, selected_device)
    generate_kwargs: dict[str, str] = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    result = asr(
        {"raw": audio, "sampling_rate": sample_rate},
        return_timestamps="word",
        chunk_length_s=chunk_length_s,
        stride_length_s=(stride_length_s, stride_length_s),
        generate_kwargs=generate_kwargs,
    )
    chunks = result.get("chunks") or []
    if chunks:
        return [
            {
                "text": chunk.get("text", "").strip(),
                "start": float(chunk["timestamp"][0]) if chunk.get("timestamp", [None])[0] is not None else None,
                "end": float(chunk["timestamp"][1]) if chunk.get("timestamp", [None, None])[1] is not None else None,
                "confidence": chunk.get("confidence"),
            }
            for chunk in chunks if chunk.get("text", "").strip()
        ]
    text = result.get("text", "").strip()
    if not text:
        raise RuntimeError(f"ASR model {model_id!r} returned no transcript")
    return [{"text": text, "start": None, "end": None, "confidence": None}]


def assign_speakers(
    utterances: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = [dict(utterance, speaker_id="unknown") for utterance in utterances]
    for utterance in output:
        start, end = utterance.get("start"), utterance.get("end")
        if start is None or end is None or end <= start:
            continue
        best_speaker = None
        best_overlap = 0.0
        for segment in diarization_segments:
            overlap = max(0.0, min(end, segment["end"]) - max(start, segment["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = segment["speaker_id"]
        if best_speaker is not None:
            utterance["speaker_id"] = best_speaker
    return output


def _resolve_diarization_device(device: str):
    import torch

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@lru_cache(maxsize=2)
def _load_diarization_pipeline(model_id: str, device: str, token: str):
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("Install pyannote.audio to run configured diarization") from exc

    model = Pipeline.from_pretrained(model_id, token=token)
    if model is None:
        raise RuntimeError(f"Could not load diarization model {model_id!r}; check access and model ID")
    model.to(_resolve_diarization_device(device))
    return model


def diarize(
    path: str | Path,
    model_id: str | None,
    token: str | None,
    device: str = "auto",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    if not model_id:
        raise RuntimeError("Set audio.diarization_model_id to enable speaker diarization")
    if not token:
        raise RuntimeError("Diarization model is configured but its Hugging Face token is missing")
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise ValueError("Set num_speakers or min/max speakers, not both")
    if min_speakers is not None and min_speakers < 1:
        raise ValueError("min_speakers must be at least 1")
    if max_speakers is not None and max_speakers < 1:
        raise ValueError("max_speakers must be at least 1")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ValueError("min_speakers cannot exceed max_speakers")
    if num_speakers is not None and num_speakers < 1:
        raise ValueError("num_speakers must be at least 1")

    model = _load_diarization_pipeline(model_id, device, token)
    kwargs: dict[str, int] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    result = model(str(path), **kwargs)
    annotation = getattr(result, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(result, "speaker_diarization", result)

    speaker_names: dict[str, str] = {}
    segments = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        original_label = str(label)
        speaker_id = speaker_names.setdefault(original_label, f"speaker_{len(speaker_names)}")
        start, end = float(segment.start), float(segment.end)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        segments.append({"start": start, "end": end, "speaker_id": speaker_id})
    if not segments:
        raise RuntimeError(f"Diarization model {model_id!r} returned no speaker segments")
    return segments
