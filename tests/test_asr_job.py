"""Process-control tests use disposable sleepers, never models or datasets."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("psutil"), "Optional ASR job-control dependency")
class AsrJobTests(unittest.TestCase):
    def test_pause_waits_for_new_checkpoint_and_resume_avoids_duplicates(self):
        import psutil
        from scripts import asr_job
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training = root / "run/training"
            training.mkdir(parents=True)
            command = [sys.executable, "-c", "import time; time.sleep(60)"]
            process = subprocess.Popen(command, **asr_job.detached_options())
            job_file = root / "job.json"
            job = {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time(),
                   "run_dir": str(root / "run"), "cwd": str(root), "command": command,
                   "log": str(root / "log.txt"), "environment": {}}
            asr_job.write(job_file, job)
            try:
                asr_job.write(root / "run/expanded-run.json", {"status": "running"})
                asr_job.write(training / "run.json", {"status": "running"})
                (training / "training.jsonl").write_text(json.dumps({"update": 10, "elapsed_seconds": 3}) + "\n")
                asr_job.request_pause(job_file)
                self.assertFalse(asr_job.pause_if_ready(job_file))
                self.assertIsNotNone(asr_job.owner(job))
                partial = training / "checkpoint.pt.partial"
                partial.write_bytes(b"test checkpoint boundary marker, not a model")
                self.assertFalse(asr_job.pause_if_ready(job_file))
                partial.replace(training / "checkpoint.pt")
                self.assertTrue(asr_job.pause_if_ready(job_file))
                self.assertIsNone(asr_job.owner(job))
                self.assertEqual(asr_job.status(job_file)["status"], "paused")
                result = asr_job.resume(job_file, watching=False)
                self.assertEqual(result["status"], "resumed")
                again = asr_job.resume(job_file, watching=False)
                self.assertEqual(again["status"], "already_running")
                self.assertEqual(result["pid"], again["pid"])
                self.assertFalse(job_file.with_suffix(".pause.json").exists())
                # After training, pause on a completed recording without marking
                # the already completed training run incomplete again.
                asr_job.write(training / "run.json", {"status": "complete"})
                asr_job.write(root / "run/experiment.json", {"stage": "fleurs_adapter"})
                asr_job.request_pause(job_file)
                (training / "checkpoint.pt").write_bytes(b"not the active ASR boundary")
                self.assertFalse(asr_job.pause_if_ready(job_file))
                progress = root / "run/fleurs-adapter/progress.jsonl"
                progress.parent.mkdir()
                progress.write_text('{"status":"complete"}\n')
                self.assertTrue(asr_job.pause_if_ready(job_file))
                self.assertEqual(asr_job.read(training / "run.json")["status"], "complete")
            finally:
                asr_job.stop_owned(asr_job.read(job_file))

    def test_changed_process_identity_cannot_be_terminated(self):
        import psutil
        from scripts import asr_job
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                   **asr_job.detached_options())
        try:
            job = {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time() - 100}
            with self.assertRaisesRegex(ValueError, "identity changed"):
                asr_job.stop_owned(job)
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
