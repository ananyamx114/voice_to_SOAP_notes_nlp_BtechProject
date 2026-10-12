from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
import logging
import os
from .audio import (
    assign_speakers,
    detect_speech,
    diarize,
    load_audio,
    transcribe_speech_segments,
)
from .datasets import load_manifest, parse_transcript
from .io_utils import append_jsonl, write_json
from .roles import assign_speaker_roles

if TYPE_CHECKING:
    from .llm import LocalLLM

logger = logging.getLogger(__name__)


def _audio_stage(
    row: dict[str, Any],
    config: dict,
    llm: LocalLLM,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audio_config = config["audio"]
    path = Path(row["audio_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found for {row['encounter_id']}: {path}")
    audio, rate = load_audio(path, int(audio_config.get("target_sample_rate", 16000)))
    duration = len(audio) / rate
    vad = audio_config.get("vad", {})
    if vad.get("enabled", True):
        speech_segments = detect_speech(
            audio,
            rate,
            threshold=float(vad.get("threshold", 0.5)),
            min_speech_duration_ms=int(vad.get("min_speech_duration_ms", 250)),
            min_silence_duration_ms=int(vad.get("min_silence_duration_ms", 100)),
            speech_pad_ms=int(vad.get("speech_pad_ms", 100)),
        )
        chunks = transcribe_speech_segments(
            audio,
            rate,
            speech_segments,
            audio_config.get("asr_model_id"),
            audio_config.get("asr_device", "auto"),
            language=audio_config.get("asr_language"),
            chunk_length_s=int(audio_config.get("asr_chunk_length_s", 30)),
            stride_length_s=int(audio_config.get("asr_stride_length_s", 5)),
        )
    else:
        speech_segments = [{"start_seconds": 0.0, "end_seconds": duration}]
        from .audio import transcribe

        chunks = transcribe(
            audio,
            rate,
            audio_config.get("asr_model_id"),
            audio_config.get("asr_device", "auto"),
            language=audio_config.get("asr_language"),
            chunk_length_s=int(audio_config.get("asr_chunk_length_s", 30)),
            stride_length_s=int(audio_config.get("asr_stride_length_s", 5)),
        )
    utterances = []
    for index, chunk in enumerate(chunks, start=1):
        utterances.append({
            "id": f"U{index}",
            "start": chunk["start"],
            "end": chunk["end"],
            "speaker_id": "unknown",
            "role": "unknown",
            "text_raw": chunk["text"],
            "text_normalized": chunk["text"],
            "confidence": chunk["confidence"],
        })
    diarization_config = audio_config.get("diarization", {})
    segments: list[dict[str, Any]] = []
    role_assignments: list[dict[str, Any]] = []
    if diarization_config.get("enabled", False):
        token = os.environ.get(audio_config.get("diarization_token_env", "HF_TOKEN"))
        segments = diarize(
            path,
            audio_config.get("diarization_model_id"),
            token,
            device=audio_config.get("diarization_device", "auto"),
            num_speakers=diarization_config.get("num_speakers"),
            min_speakers=diarization_config.get("min_speakers"),
            max_speakers=diarization_config.get("max_speakers"),
        )
        utterances = assign_speakers(utterances, segments)
        role_config = audio_config.get("role_assignment", {})
        if role_config.get("enabled", True):
            utterances, role_assignments = assign_speaker_roles(llm, utterances)
    return utterances, {
        "audio_path": str(path),
        "sample_rate": rate,
        "duration_seconds": duration,
        "vad_enabled": bool(vad.get("enabled", True)),
        "speech_segments": speech_segments,
        "diarization_enabled": bool(diarization_config.get("enabled", False)),
        "diarization_segments": segments,
        "role_assignments": role_assignments,
    }


def _translate_plan(llm: LocalLLM, note: dict, language: str) -> list[dict[str, str]]:
    plan = "\n".join(item["text"] for item in note["plan"])
    if not plan:
        return []
    result = llm.complete_json(
        "Translate clinical instructions accurately. Preserve drug names, numbers, dose, and frequency. Return JSON only.",
        f'Translate this plan to {language}. Return {{"translation":"..."}}.\n\n{plan}',
    )
    translated = result.get("translation")
    if not isinstance(translated, str) or not translated.strip():
        raise ValueError("Plan translation response is missing translation text")
    return [{"language": language, "text": translated}]


def run(config: dict, manifest_path: str, output_dir: str) -> None:
    from .llm import LocalLLM
    from .normalize import Normalizer
    from .entity_linking import TerminologyLinker
    from .steps import extract_facts, generate_verified_soap, render_soap

    rows = load_manifest(manifest_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    llm = LocalLLM(config["llm"])
    normalizer = Normalizer(config["normalization"]["lookup_dir"])
    entity_linking_config = config.get("entity_linking", {})
    terminology_linker = (
        TerminologyLinker(entity_linking_config["terminology_path"])
        if entity_linking_config.get("enabled", True)
        else None
    )
    soap_generation_config = config.get("soap_generation", {})
    soap_backend = soap_generation_config.get("backend", "local_llm")
    if soap_backend == "lora":
        from .soap_lora import LoRASOAPGenerator

        soap_generator = LoRASOAPGenerator(
            soap_generation_config.get("base_model"),
            soap_generation_config.get("adapter_path"),
            max_new_tokens=int(soap_generation_config.get("max_new_tokens", 512)),
        )
    elif soap_backend == "local_llm":
        soap_generator = None
    else:
        raise ValueError(f"Unsupported SOAP generation backend {soap_backend!r}")
    status_path = out_root / "run.jsonl"

    for row in rows:
        encounter_id = row["encounter_id"]
        encounter_dir = out_root / encounter_id
        encounter_dir.mkdir(parents=True, exist_ok=True)
        try:
            if row["transcript"]:
                utterances = parse_transcript(row["transcript"])
                audio_metadata = {"source": "provided_transcript"}
            else:
                utterances, audio_metadata = _audio_stage(row, config, llm)
            write_json(encounter_dir / "audio.json", audio_metadata)
            write_json(encounter_dir / "transcript_raw.json", utterances)

            utterances, changes = normalizer.normalize(utterances)
            write_json(encounter_dir / "transcript.json", {"utterances": utterances, "changes": changes})

            facts = extract_facts(llm, utterances)
            if terminology_linker is not None:
                facts = terminology_linker.link_facts(facts)
            write_json(encounter_dir / "facts.json", {"facts": facts})
            verification_config = config.get("verification", {})
            outcome = generate_verified_soap(
                llm,
                utterances,
                facts,
                normalizer,
                soap_generator=soap_generator,
                max_attempts=int(verification_config.get("max_attempts", 2)),
            )
            checked = outcome["claims"]
            supported = outcome["supported_claims"]
            write_json(encounter_dir / "soap_draft.json", render_soap(checked))
            verified = {
                "soap": render_soap(checked, supported_only=True),
                "claims": checked,
                "omissions": outcome["omissions"],
                "warnings": outcome["warnings"],
                "verification_attempts": outcome["attempts"],
                "review_required": outcome["review_required"],
            }
            translation = config.get("translation", {})
            if translation.get("enabled"):
                target = translation.get("target_language")
                if target not in {"hi", "mr"}:
                    raise ValueError("translation.target_language must be 'hi' or 'mr'")
                verified["translated_plan"] = _translate_plan(llm, verified["soap"], target)
            write_json(encounter_dir / "soap_verified.json", verified)
            append_jsonl(status_path, {"encounter_id": encounter_id, "status": "complete"})
        except Exception as exc:
            logger.exception("Encounter %s failed", encounter_id)
            append_jsonl(status_path, {"encounter_id": encounter_id, "status": "failed", "error": str(exc)})
            if config.get("pipeline", {}).get("stop_on_error", False):
                raise
