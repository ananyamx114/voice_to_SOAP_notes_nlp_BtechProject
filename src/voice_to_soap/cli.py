import argparse
import json
import logging
import os
from pathlib import Path
import soundfile as sf
from .audio import (
    assign_speakers,
    detect_speech,
    diarize,
    load_audio,
    prepare_audio_file,
    save_speech_segments,
    transcribe,
)
from .config import load_config
from .datasets import load_manifest
from .io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(prog="voice-to-soap")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare-audio", help="convert manifest audio to mono WAV at a common sample rate"
    )
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--outdir", default="data/processed/audio")
    prepare_parser.add_argument("--sample-rate", type=int, default=16000)

    vad_parser = commands.add_parser(
        "detect-speech", help="find and save speech-only segments from prepared audio"
    )
    vad_parser.add_argument("--manifest", required=True)
    vad_parser.add_argument("--audio-dir", default="data/processed/audio")
    vad_parser.add_argument("--outdir", default="data/processed/speech")
    vad_parser.add_argument("--sample-rate", type=int, default=16000)
    vad_parser.add_argument("--threshold", type=float, default=0.5)
    vad_parser.add_argument("--min-speech-ms", type=int, default=250)
    vad_parser.add_argument("--min-silence-ms", type=int, default=100)
    vad_parser.add_argument("--padding-ms", type=int, default=100)

    asr_parser = commands.add_parser(
        "transcribe-audio", help="transcribe one recording with the configured ASR model"
    )
    asr_parser.add_argument("--audio", required=True)
    asr_parser.add_argument("--out", required=True)
    asr_parser.add_argument("--config", default="configs/base.yaml")
    asr_parser.add_argument("--model", default=None, help="Override audio.asr_model_id")
    asr_parser.add_argument("--device", default=None, help="auto, cuda:0, mps, or cpu")
    asr_parser.add_argument("--language", default=None, help="Optional language name, e.g. marathi")

    diarization_parser = commands.add_parser(
        "diarize-audio", help="identify speaker turns in one recording"
    )
    diarization_parser.add_argument("--audio", required=True)
    diarization_parser.add_argument("--out", required=True)
    diarization_parser.add_argument("--config", default="configs/base.yaml")
    diarization_parser.add_argument("--device", default=None)
    diarization_parser.add_argument("--num-speakers", type=int, default=None)

    diarization_manifest_parser = commands.add_parser(
        "diarize-manifest", help="diarize audio files listed in a manifest"
    )
    diarization_manifest_parser.add_argument("--manifest", required=True)
    diarization_manifest_parser.add_argument("--out", required=True)
    diarization_manifest_parser.add_argument("--config", default="configs/base.yaml")
    diarization_manifest_parser.add_argument("--device", default=None)
    diarization_manifest_parser.add_argument("--num-speakers", type=int, default=None)
    diarization_manifest_parser.add_argument("--languages", default="en,hi")
    diarization_manifest_parser.add_argument("--limit-per-language", type=int, default=None)

    run_parser = commands.add_parser("run", help="run encounters in a manifest")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--config", default="configs/base.yaml")
    run_parser.add_argument("--outdir", default="data/outputs/run")

    eka_import_parser = commands.add_parser(
        "import-eka-asr",
        help="export Eka Parquet audio and references into an ASR evaluation manifest",
    )
    eka_import_parser.add_argument(
        "--dataset-dir", default="data/raw/eka-medical-asr-evaluation"
    )
    eka_import_parser.add_argument(
        "--outdir", default="data/processed/eka-medical-asr-evaluation"
    )
    eka_import_parser.add_argument(
        "--manifest-out", default="data/manifests/eka-medical-asr-evaluation.csv"
    )
    eka_import_parser.add_argument("--languages", default="en,hi")
    eka_import_parser.add_argument("--limit-per-language", type=int, default=None)
    eka_import_parser.add_argument(
        "--overwrite", action="store_true", help="overwrite exports with matching filenames"
    )

    asr_eval_parser = commands.add_parser(
        "evaluate-asr",
        help="transcribe manifest audio and score it against reference transcripts",
    )
    asr_eval_parser.add_argument("--manifest", required=True)
    asr_eval_parser.add_argument("--config", default="configs/base.yaml")
    asr_eval_parser.add_argument("--model", default=None, help="Override audio.asr_model_id")
    asr_eval_parser.add_argument("--device", default=None, help="auto, cuda:0, mps, or cpu")
    asr_eval_parser.add_argument(
        "--outdir", default="data/outputs/eka-medical-asr-evaluation"
    )
    asr_eval_parser.add_argument("--languages", default="en,hi")
    asr_eval_parser.add_argument("--limit-per-language", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "diarize-audio":
        config = load_config(args.config)
        audio_config = config["audio"]
        token = os.environ.get(audio_config.get("diarization_token_env", "HF_TOKEN"))
        diarization_settings = audio_config.get("diarization", {})
        segments = diarize(
            args.audio,
            audio_config.get("diarization_model_id"),
            token,
            device=args.device or audio_config.get("diarization_device", "auto"),
            num_speakers=(
                args.num_speakers
                if args.num_speakers is not None
                else diarization_settings.get("num_speakers")
            ),
            min_speakers=diarization_settings.get("min_speakers"),
            max_speakers=diarization_settings.get("max_speakers"),
        )
        write_json(args.out, {
            "audio_path": str(Path(args.audio).resolve()),
            "model_id": audio_config.get("diarization_model_id"),
            "segments": segments,
        })
        logging.info("Wrote diarization segments to %s", args.out)
    elif args.command == "diarize-manifest":
        config = load_config(args.config)
        audio_config = config["audio"]
        token = os.environ.get(audio_config.get("diarization_token_env", "HF_TOKEN"))
        languages = {
            language.strip().lower()
            for language in args.languages.split(",")
            if language.strip()
        }
        if not languages:
            raise ValueError("Select at least one language for manifest diarization")
        if args.limit_per_language is not None and args.limit_per_language < 1:
            raise ValueError("limit-per-language must be positive when provided")

        rows = load_manifest(args.manifest)
        selected = []
        seen_by_language: dict[str, int] = {}
        for row in rows:
            language = row.get("language", "").strip().lower()
            if language not in languages:
                continue
            if (
                args.limit_per_language is not None
                and seen_by_language.get(language, 0) >= args.limit_per_language
            ):
                continue
            if not row.get("audio_path"):
                raise ValueError(
                    f"Encounter {row['encounter_id']!r} has no audio_path for diarization"
                )
            selected.append(row)
            seen_by_language[language] = seen_by_language.get(language, 0) + 1
        if not selected:
            raise ValueError("No manifest examples matched the selected languages")

        diarization_settings = audio_config.get("diarization", {})
        results = []
        for row in selected:
            segments = diarize(
                row["audio_path"],
                audio_config.get("diarization_model_id"),
                token,
                device=args.device or audio_config.get("diarization_device", "auto"),
                num_speakers=(
                    args.num_speakers
                    if args.num_speakers is not None
                    else diarization_settings.get("num_speakers")
                ),
                min_speakers=diarization_settings.get("min_speakers"),
                max_speakers=diarization_settings.get("max_speakers"),
            )
            results.append({
                "encounter_id": row["encounter_id"],
                "language": row.get("language", ""),
                "audio_path": str(Path(row["audio_path"]).resolve()),
                "model_id": audio_config.get("diarization_model_id"),
                "segments": segments,
            })

        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_output.open("w", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        temporary_output.replace(output_path)
        logging.info(
            "Diarized %d manifest examples; output saved to %s",
            len(results),
            output_path,
        )
    elif args.command == "transcribe-audio":
        config = load_config(args.config)
        audio_config = config["audio"]
        model_id = args.model or audio_config.get("asr_model_id")
        waveform, sample_rate = load_audio(
            args.audio, int(audio_config.get("target_sample_rate", 16000))
        )
        chunks = transcribe(
            waveform,
            sample_rate,
            model_id,
            args.device or audio_config.get("asr_device", "auto"),
            language=args.language or audio_config.get("asr_language"),
            chunk_length_s=int(audio_config.get("asr_chunk_length_s", 30)),
            stride_length_s=int(audio_config.get("asr_stride_length_s", 5)),
        )
        write_json(args.out, {
            "audio_path": str(Path(args.audio).resolve()),
            "model_id": model_id,
            "sample_rate": sample_rate,
            "text": " ".join(chunk["text"] for chunk in chunks).strip(),
            "chunks": chunks,
        })
        logging.info("Wrote ASR transcript to %s", args.out)
    elif args.command == "prepare-audio":
        output_root = Path(args.outdir)
        rows = load_manifest(args.manifest)
        failed = []
        for row in rows:
            if not row["audio_path"]:
                logging.info("Skipping %s: manifest has no audio path", row["encounter_id"])
                continue
            encounter_dir = output_root / row["encounter_id"]
            try:
                metadata = prepare_audio_file(
                    row["audio_path"],
                    encounter_dir / "audio_mono.wav",
                    target_rate=args.sample_rate,
                )
                write_json(encounter_dir / "audio_metadata.json", metadata)
                logging.info("Prepared audio for %s", row["encounter_id"])
            except Exception as exc:
                failed.append((row["encounter_id"], str(exc)))
                logging.exception("Audio preparation failed for %s", row["encounter_id"])
        if failed:
            details = "; ".join(f"{encounter_id}: {error}" for encounter_id, error in failed)
            raise RuntimeError(f"Audio preparation failed for {len(failed)} encounter(s): {details}")
    elif args.command == "detect-speech":
        audio_root = Path(args.audio_dir)
        output_root = Path(args.outdir)
        rows = load_manifest(args.manifest)
        failed = []
        for row in rows:
            if not row["audio_path"]:
                logging.info("Skipping %s: manifest has no audio path", row["encounter_id"])
                continue
            encounter_id = row["encounter_id"]
            prepared_path = audio_root / encounter_id / "audio_mono.wav"
            try:
                info = sf.info(str(prepared_path))
                if info.channels != 1 or info.samplerate != args.sample_rate:
                    raise ValueError(
                        f"Expected mono {args.sample_rate} Hz prepared audio, got "
                        f"{info.channels} channel(s) at {info.samplerate} Hz"
                    )
                waveform, rate = load_audio(prepared_path, args.sample_rate)
                segments = detect_speech(
                    waveform,
                    rate,
                    threshold=args.threshold,
                    min_speech_duration_ms=args.min_speech_ms,
                    min_silence_duration_ms=args.min_silence_ms,
                    speech_pad_ms=args.padding_ms,
                )
                encounter_output = output_root / encounter_id
                saved = save_speech_segments(
                    waveform, rate, segments, encounter_output / "segments"
                )
                write_json(encounter_output / "speech_segments.json", {
                    "encounter_id": encounter_id,
                    "source_audio": str(prepared_path.resolve()),
                    "sample_rate": rate,
                    "speech_detected": bool(saved),
                    "segments": saved,
                })
                logging.info("Detected %d speech segment(s) for %s", len(saved), encounter_id)
            except Exception as exc:
                failed.append((encounter_id, str(exc)))
                logging.exception("Speech detection failed for %s", encounter_id)
        if failed:
            details = "; ".join(f"{encounter_id}: {error}" for encounter_id, error in failed)
            raise RuntimeError(f"Speech detection failed for {len(failed)} encounter(s): {details}")
    elif args.command == "run":
        from .pipeline import run

        config = load_config(args.config)
        run(config, args.manifest, args.outdir)
    elif args.command == "import-eka-asr":
        from .asr_datasets import import_eka_asr_dataset

        languages = [language for language in args.languages.split(",") if language.strip()]
        result = import_eka_asr_dataset(
            args.dataset_dir,
            args.outdir,
            manifest_path=args.manifest_out,
            languages=languages,
            limit_per_language=args.limit_per_language,
            overwrite=args.overwrite,
        )
        logging.info(
            "Exported %d Eka ASR test examples to %s",
            result["total_examples"],
            result["manifest_path"],
        )
    elif args.command == "evaluate-asr":
        from .asr_datasets import evaluate_asr_manifest

        config = load_config(args.config)
        audio_config = config["audio"]
        model_id = args.model or audio_config.get("asr_model_id")
        if not model_id:
            raise ValueError("Set audio.asr_model_id or pass --model")
        languages = [language for language in args.languages.split(",") if language.strip()]
        result = evaluate_asr_manifest(
            args.manifest,
            model_id=model_id,
            output_dir=args.outdir,
            device=args.device or audio_config.get("asr_device", "auto"),
            sample_rate=int(audio_config.get("target_sample_rate", 16000)),
            languages=languages,
            limit_per_language=args.limit_per_language,
            chunk_length_s=int(audio_config.get("asr_chunk_length_s", 30)),
            stride_length_s=int(audio_config.get("asr_stride_length_s", 5)),
        )
        logging.info(
            "Evaluated %d Eka ASR examples with %s; metrics saved to %s",
            result["examples"],
            model_id,
            Path(args.outdir) / "metrics.json",
        )


if __name__ == "__main__":
    main()
