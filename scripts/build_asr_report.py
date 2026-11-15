"""Export aggregate ASR evidence and a report section from verified run archives.

No audio, transcripts, predictions, credentials or model weights are exported.
Incomplete evaluation is explicitly marked pending; --require-complete refuses it.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tarfile
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

TRAIN = "data/outputs/asr-lora-experiment-run2/training/"
EVAL = "data/outputs/asr-lora-kaggle-evaluation/"
LANGUAGES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
EXPECTED_COUNTS = {"eka": 4, "betrac": 2, "sully": 1, "fleurs": 64}
EXPECTED_LANGUAGES = {"eka": {"en": 2, "hi": 2}, "betrac": {"en": 2},
                      "sully": {"en": 1}, "fleurs": {"hi": 32, "mr": 32}}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_archive(path):
    path = Path(path)
    inventory = json.loads(path.with_suffix(".inventory.json").read_text())
    if digest(path) != inventory["sha256"]:
        raise ValueError("Report input archive checksum mismatch")
    expected = {row["path"]: row for row in inventory["files"]}
    records = {}
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Report input archive inventory mismatch")
        for member in members:
            if not member.isfile():
                raise ValueError("Unexpected report archive member type")
            # JSON aggregates are sufficient; never load or extract checkpoint tensors.
            wanted = (member.name.startswith(TRAIN) and member.name.endswith(".json")) or (
                member.name.startswith(EVAL) and (member.name.endswith("/experiment.json") or
                                                 member.name.endswith("-comparison.json") or
                                                 member.name.endswith("/asr.json")))
            if wanted:
                payload = archive.extractfile(member).read()
                if hashlib.sha256(payload).hexdigest() != expected[member.name]["sha256"]:
                    raise ValueError("Report JSON checksum mismatch")
                records[member.name] = json.loads(payload)
    return records, inventory["sha256"]


def assemble(training_records, evaluation_records=None, require_complete=False):
    training = training_records[TRAIN + "run.json"]
    if training["status"] != "complete" or training["optimizer_updates"] != 515:
        raise ValueError("The report requires the completed five-epoch training run")
    adapter = training_records[TRAIN + "best-adapter/asr_adapter.json"]
    if training["model_id"] != "openai/whisper-large-v3" or training["config"]["training"]["epochs"] != 5:
        raise ValueError("Unexpected model or training schedule for this report")
    if adapter["status"] != "complete" or adapter["config"] != training["config"]:
        raise ValueError("Adapter and training configuration differ")
    epochs = []
    for epoch in range(1, 6):
        metrics = training_records[TRAIN + f"epoch-{epoch}/metrics.json"]
        epochs.append({"epoch": epoch, **metrics,
                       "macro_language_wer": sum(v["wer"] for v in metrics["by_language"].values()) / 3})
    selected = min(epochs, key=lambda row: row["macro_language_wer"])
    if adapter["epoch"] != selected["epoch"] or adapter["validation"] != selected_metrics(selected):
        raise ValueError("Selected adapter does not match validation-only checkpoint selection")
    records = evaluation_records or training_records
    experiment = records.get(EVAL + "experiment.json", {})
    complete = experiment.get("status") == "complete"
    if require_complete and not complete:
        raise ValueError("Paired CUDA evaluation is incomplete; refusing a final report")
    comparisons = {}
    rows = []
    for dataset, count in EXPECTED_COUNTS.items():
        value = records.get(EVAL + f"{dataset}-comparison.json")
        if value is None:
            if complete:
                raise ValueError(f"Missing final paired comparison: {dataset}")
            continue
        if value["encounters"] != count or len(value["paired_ids"]) != count:
            raise ValueError("Unexpected paired evaluation coverage")
        if sum(v["base"]["examples"] for v in value["by_language"].values()) != count:
            raise ValueError("Paired language counts differ from encounter coverage")
        if {key: item["base"]["examples"] for key, item in value["by_language"].items()} != EXPECTED_LANGUAGES[dataset]:
            raise ValueError("Paired evaluation language coverage differs from the frozen cohort")
        for uid in value["paired_ids"]:
            before = records[EVAL + f"{dataset}-base/{uid}/asr.json"]
            after = records[EVAL + f"{dataset}-adapter/{uid}/asr.json"]
            if before["status"] != "complete" or after["status"] != "complete":
                raise ValueError("Do not report failed/incomplete ASR recordings")
            if after["model"]["adapter"]["sha256"] != adapter["adapter_sha256"]:
                raise ValueError("Evaluated adapter differs from validation-selected weights")
            if before["model"].get("adapter") or before["model"]["model_id"] != training["model_id"] or before["model"]["revision"] != training["revision"]:
                raise ValueError("Paired base does not match the frozen training checkpoint")
            for key in ("audio_sha256", "reference_transcript", "language", "decoded_regions", "seed"):
                if before[key] != after[key]:
                    raise ValueError(f"Report paired recording mismatch: {key}")
            for key in ("model_id", "revision", "precision", "device", "generation", "algorithm", "timestamps", "implementation_sha256"):
                if before["model"].get(key) != after["model"].get(key):
                    raise ValueError(f"Report paired decoder mismatch: {key}")
        comparisons[dataset] = value
        for language, metrics in value["by_language"].items():
            if metrics["base"]["examples"] != metrics["adapter"]["examples"]:
                raise ValueError("Paired base/adapter counts differ")
            rows.append({"dataset": dataset, "language": language,
                         "recordings": metrics["base"]["examples"],
                         "base_wer": metrics["base"]["wer"], "adapter_wer": metrics["adapter"]["wer"],
                         "wer_change_pp": 100 * (metrics["adapter"]["wer"] - metrics["base"]["wer"]),
                         "base_cer": metrics["base"]["cer"], "adapter_cer": metrics["adapter"]["cer"]})
    audit = experiment.get("audit", training["data_audit"])
    if audit["train_segments"] != 1644 or audit["validation_segments"] != 239:
        raise ValueError("Unexpected cohort for the expanded-run report")
    return {"status": "complete" if complete else "validation_complete_test_pending",
            "model_id": training["model_id"], "revision": training["revision"],
            "config": training["config"], "parameters": training["parameters"],
            "optimizer_updates": training["optimizer_updates"], "versions": training["versions"],
            "base_precision": training["base_precision"], "quantization": training["quantization"],
            "training_and_validation_seconds_cuda": training["elapsed_seconds"],
            "adapter_sha256": adapter["adapter_sha256"], "selected_epoch": adapter["epoch"],
            "audit": audit, "training_by_language": experiment.get("training_by_language", {}),
            "validation_base_historical_mps": adapter["baseline_validation"],
            "validation_selected_cuda": adapter["validation"], "validation_epochs": epochs,
            "paired_cuda_rows": rows, "paired_cuda_comparisons": comparisons,
            "paired_recordings": sum(r["recordings"] for r in rows),
            "wer_improved_groups": sum(r["adapter_wer"] < r["base_wer"] for r in rows),
            "wer_regressed_groups": sum(r["adapter_wer"] > r["base_wer"] for r in rows),
            "cer_regressed_groups": sum(r["adapter_cer"] > r["base_cer"] for r in rows)}


def selected_metrics(row):
    return {key: value for key, value in row.items() if key not in {"epoch", "macro_language_wer"}}


def markdown(summary):
    s = summary
    lines = ["# ASR: Whisper Large-v3 with LoRA", "",
             f"Evidence status: **{s['status']}**. Training finished all five epochs / 515 optimizer updates.", "",
             "## Method", "",
             "The audio-to-text component uses the complete pretrained Whisper Large-v3 encoder-decoder. "
             "The frozen base remains FP16 without quantization. LoRA adapts the query and value projections "
             "in encoder self-attention, decoder self-attention and decoder cross-attention. "
             "This is local model adaptation, rather than a hosted language-model transcription service.", "",
             f"The run trains {s['parameters']['trainable_parameters']:,} parameters out of "
             f"{s['parameters']['total_parameters']:,} total parameters. "
             "LoRA rank is 16, alpha 32 and dropout 0.05. AdamW uses learning rate 5e-5, weight decay 0.01, "
             "gradient accumulation 16, warmup ratio 0.1 and gradient-norm clipping at 1.0. "
             "The seed is 42; microbatch size is one. Adapter weights remain FP32, with gradient scaling "
             "and non-reentrant gradient checkpointing.", "",
             f"Base checkpoint: `{s['model_id']}@{s['revision']}`. "
             f"Selected adapter SHA-256: `{s['adapter_sha256']}`.", "",
             "## Data and protocol", "",
             "Training uses 1,644 aligned segments: 1,045 English BeTraC segments from 36 consultations (7,222.37 seconds), "
             "321 Hindi FLEURS clips (3,641.28 seconds), and 278 Marathi FLEURS clips (3,631.14 seconds). "
             "This totals 4.03 hours. Validation contains 239 segments: 175 English, 32 Hindi and 32 Marathi. "
             "BeTraC English speech is synthetic; FLEURS is general-domain speech. "
             "These data do not establish clinical performance in Hindi or Marathi.", "",
             "Inputs are 16 kHz mono. English segment boundaries use CTC forced alignment with rejection "
             "thresholds; alignment is not human-reviewed ground truth. FLEURS supervision uses the "
             "dataset's complete audio/reference pairs. Targets are checked without truncation. "
             "Training and validation IDs, encounter groups, source-audio hashes and full-encounter text "
             "are checked for overlap. Eka test recordings never enter training.", "",
             "Checkpoint selection uses the equally weighted mean of English, Hindi and Marathi validation "
             f"WER; epoch {s['selected_epoch']} was selected. Validation uses greedy short-segment decoding. "
             "Production evaluation uses native sequential decoding with temperature fallback and word "
             "timestamps. Within each paired test, base and adapter must match audio hashes, references, "
             "language hints, seeds, decoded regions, model revision, precision, device and decoding settings.", "",
             "WER = (substitutions + deletions + insertions) / reference words; CER applies the same edit "
             "calculation to characters. Scoring case-folds text, removes Unicode punctuation and collapses "
             "whitespace. Aggregate WER is calculated over reference/hypothesis sequences, not by averaging "
             "per-recording percentages. Lower values are better.", "",
             "## Validation results", "",
             "The baseline validation used MPS; adapted validation used CUDA after checkpoint migration. "
             "This table is diagnostic validation evidence, not a same-backend paired test comparison.", "",
             "| Language | Segments | Historical base WER | Selected WER | Historical base CER | Selected CER |",
             "|---|---:|---:|---:|---:|---:|"]
    for language, metrics in s["validation_selected_cuda"]["by_language"].items():
        base = s["validation_base_historical_mps"]["by_language"][language]
        lines.append(f"| {LANGUAGES[language]} | {metrics['examples']} | {base['wer']*100:.2f}% | "
                     f"{metrics['wer']*100:.2f}% | {base['cer']*100:.2f}% | {metrics['cer']*100:.2f}% |")
    lines += ["", "## Paired CUDA evaluation", ""]
    if s["status"] != "complete":
        lines += ["**PENDING: evaluation is still running. No full test conclusion is available.**", ""]
    lines += [f"Measured coverage currently available: {s['paired_recordings']}/71 selected recordings. "
              "The planned coverage is Eka 4, BeTraC 2, Sully/PriMock57 1 and FLEURS 64 (32 Hindi, 32 Marathi). "
              "This is selected-recording evaluation, not full-corpus benchmarking.", "",
              "| Dataset | Language | Recordings | Base WER | LoRA WER | Change (pp) | Base CER | LoRA CER |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in s["paired_cuda_rows"]:
        lines.append(f"| {row['dataset']} | {LANGUAGES[row['language']]} | {row['recordings']} | "
                     f"{row['base_wer']*100:.2f}% | {row['adapter_wer']*100:.2f}% | {row['wer_change_pp']:+.2f} | "
                     f"{row['base_cer']*100:.2f}% | {row['adapter_cer']*100:.2f}% |")
    if s["paired_cuda_rows"]:
        ties = len(s["paired_cuda_rows"]) - s["wer_improved_groups"] - s["wer_regressed_groups"]
        lines += ["", f"Across the available dataset/language groups, WER improved in {s['wer_improved_groups']}, "
                  f"regressed in {s['wer_regressed_groups']} and tied in {ties}. "
                  f"CER regressed in {s['cer_regressed_groups']} groups. Negative percentage-point change means lower WER."]
    if s["status"] == "complete":
        english = [r for r in s["paired_cuda_rows"] if r["language"] == "en"]
        indic = [r for r in s["paired_cuda_rows"] if r["language"] in {"hi", "mr"}]
        if english and all(r["adapter_wer"] > r["base_wer"] for r in english) and indic and all(r["adapter_wer"] < r["base_wer"] for r in indic):
            lines += ["", "The selected shared adapter improved Hindi on Eka and FLEURS and Marathi on FLEURS, "
                      "while all three English diagnostic groups regressed. The experiment therefore supports "
                      "a multilingual adaptation tradeoff, rather than universal ASR improvement. The selected "
                      "adapter was not changed after observing the test results."]
    lines += ["", "## Discussion and limitations", "",
              "The validation gains show adaptation to the selected development distribution; they do not "
              "guarantee generalization to other microphones, accents or clinical conversations. The known "
              "Eka recordings are diagnostic because they have been inspected in earlier experiments. "
              "A two-recording language subset is too small for broad accuracy claims. Marathi validation "
              "WER remains high. High validation improvement alongside a test regression requires error "
              "analysis; it does not by itself prove overfitting or underfitting.", "",
              "The local MPS run resumed on one NVIDIA T4 using preserved adapter, AdamW, scaler, CPU RNG "
              "and training position. CUDA RNG was explicitly reseeded because MPS RNG cannot be translated. "
              "PyTorch/NumPy platform differences were recorded; bitwise-equivalent continuation is not claimed. "
              "A post-training coordinator interpreter error was corrected without retraining or deleting checkpoints.", "",
              "WER/CER measure transcription overlap, not clinical correctness. Medication names, quantities, "
              "negation, omissions and timestamp boundaries need separate review. Numeric/negation inventory "
              "checks, when used, are lexical flags rather than clinical validation. Every downstream SOAP note "
              "still requires clinician approval. This ASR experiment does not measure Qwen SOAP quality or "
              "end-to-end clinical performance. Sully contains overlapping ACI-Bench material elsewhere in "
              "the collection; do not call overlapping encounters independent end-to-end test evidence.", "",
              "The remaining existing training data can support a later experiment. Keep this run frozen, "
              "exclude validation/test overlap, use reference transcripts and vary data coverage in a separately "
              "recorded experiment. Repeated test-guided tuning requires an additional untouched final test set.", "",
              "## Reproducibility and attribution", "",
              "Configuration: `configs/asr-lora-expanded.yaml`. Source checkpoints and predictions remain "
              "ignored local/private artifacts. The summary records input archive hashes, model pin, adapter "
              "hash and software versions. Tables contain measured aggregate values only; no transcripts, "
              "audio, weights, credentials or machine-specific runtime settings are included.", "",
              "Dataset pins: BeTraC `96f7b94e662dac0d1394cfdf55e8b8ae81422663`; "
              "FLEURS converted parquet `168de341b3db6859a9bac1c50a2ef5e3b47647e0`, "
              "with dataset card `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.", "",
              "- Whisper Large-v3 model card: https://huggingface.co/openai/whisper-large-v3",
              "- Hu et al., LoRA: https://arxiv.org/abs/2106.09685",
              "- BeTraC dataset and CC BY 4.0 attribution: https://huggingface.co/datasets/BeTraC/betrac-2026",
              "- FLEURS dataset and CC BY 4.0 attribution: https://huggingface.co/datasets/google/fleurs",
              "- Eka and Sully retain their upstream terms; the selected source recordings are not redistributed."]
    return "\n".join(lines) + "\n"


def figures(summary, outdir):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".local/matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".local/cache").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    for language, label in LANGUAGES.items():
        ax.plot([r["epoch"] for r in summary["validation_epochs"]],
                [100*r["by_language"][language]["wer"] for r in summary["validation_epochs"]],
                marker="o", label=label)
    ax.set(xlabel="Epoch", ylabel="Validation WER (%)", title="Whisper Large-v3 LoRA: validation across five epochs", xticks=range(1, 6))
    ax.legend()
    ax.grid(alpha=.2)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(outdir / f"validation-wer.{extension}", dpi=300)
    clean_svg(outdir / "validation-wer.svg")
    plt.close(fig)
    rows = summary["paired_cuda_rows"]
    if rows:
        fig, ax = plt.subplots(figsize=(8.4, 4.3), layout="constrained")
        x = list(range(len(rows)))
        ax.bar([v-.2 for v in x], [100*r["base_wer"] for r in rows], width=.4, label="Base")
        ax.bar([v+.2 for v in x], [100*r["adapter_wer"] for r in rows], width=.4, label="LoRA")
        ax.set_xticks(x, [f"{r['dataset']}\n{r['language']} (n={r['recordings']})" for r in rows])
        ax.set(ylabel="WER (%)", title="Matched CUDA evaluation on selected recordings")
        ax.legend()
        ax.grid(axis="y", alpha=.2)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(outdir / f"paired-test-wer.{extension}", dpi=300)
        clean_svg(outdir / "paired-test-wer.svg")
        plt.close(fig)


def clean_svg(path):
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def word_document(text, outdir):
    # Minimal standard OOXML document, using only the Python standard library.
    # Figures are exported separately for insertion into the team's template.
    body = []
    def paragraph(value, style=None):
        properties = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        value = escape(value.replace("**", "").replace("`", ""))
        return f'<w:p>{properties}<w:r><w:t xml:space="preserve">{value}</w:t></w:r></w:p>'
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("|"):
            table_rows = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [c.strip() for c in lines[index].strip("|").split("|")]
                if not all(set(c) <= {"-", ":"} for c in cells):
                    table_rows.append(cells)
                index += 1
            table = ['<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/></w:tblPr>']
            for row in table_rows:
                table.append('<w:tr>' + ''.join('<w:tc>' + paragraph(value) + '</w:tc>' for value in row) + '</w:tr>')
            table.append('</w:tbl>')
            body.append(''.join(table))
            continue
        if line.startswith("# "):
            body.append(paragraph(line[2:], "Title"))
        elif line.startswith("## "):
            body.append(paragraph(line[3:], "Heading1"))
        elif line:
            body.append(paragraph(line))
        index += 1
    namespace = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    document = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{namespace}"><w:body>' + ''.join(body) + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="850" w:bottom="1134" w:left="850"/></w:sectPr></w:body></w:document>'
    content_types = '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
    relationships = 'http://schemas.openxmlformats.org/package/2006/relationships'
    office = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    styles = f'<w:styles xmlns:w="{namespace}"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style></w:styles>'
    with ZipFile(outdir / "ASR_REPORT_SECTION.docx", "w", ZIP_DEFLATED) as output:
        output.writestr("[Content_Types].xml", content_types)
        output.writestr("_rels/.rels", f'<Relationships xmlns="{relationships}"><Relationship Id="rId1" Type="{office}/officeDocument" Target="word/document.xml"/></Relationships>')
        output.writestr("word/document.xml", document)
        output.writestr("word/styles.xml", styles)
        output.writestr("word/_rels/document.xml.rels", f'<Relationships xmlns="{relationships}"><Relationship Id="rId1" Type="{office}/styles" Target="styles.xml"/></Relationships>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-archive", required=True)
    parser.add_argument("--evaluation-archive")
    parser.add_argument("--outdir", default="results/asr-lora-run2")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--docx", action="store_true")
    args = parser.parse_args()
    training, training_sha = read_archive(args.training_archive)
    evaluation, evaluation_sha = read_archive(args.evaluation_archive) if args.evaluation_archive else (None, None)
    summary = assemble(training, evaluation, args.require_complete)
    summary["source_archive_sha256"] = {"training": training_sha, "evaluation": evaluation_sha}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = markdown(summary)
    (outdir / "ASR_REPORT_SECTION.md").write_text(text, encoding="utf-8")
    (outdir / "README.md").write_text(text, encoding="utf-8")
    epoch_rows = [{"epoch": row["epoch"], "language": language, "wer": metrics["wer"], "cer": metrics["cer"]}
                  for row in summary["validation_epochs"] for language, metrics in row["by_language"].items()]
    for name, rows in (("validation-epochs.csv", epoch_rows), ("paired-test-results.csv", summary["paired_cuda_rows"])):
        with (outdir / name).open("w", newline="", encoding="utf-8") as stream:
            columns = list(rows[0]) if rows else ["dataset", "language", "recordings", "base_wer", "adapter_wer", "wer_change_pp", "base_cer", "adapter_cer"]
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    if args.plots:
        figures(summary, outdir)
    if args.docx:
        word_document(text, outdir)
    print(json.dumps({"status": summary["status"], "outdir": str(outdir),
                      "paired_recordings": summary["paired_recordings"]}, indent=2))


if __name__ == "__main__":
    main()
