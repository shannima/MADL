import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseHygieneTests(unittest.TestCase):
    def test_runtime_sources_have_no_machine_local_paths_or_obsolete_taxonomy(self):
        forbidden = ("/mnt/", "/root/", "tmp_export", "SIDACls", "E:\\", "D:\\", "F:\\")
        for relative_root in ("src", "scripts", "configs"):
            for path in (ROOT / relative_root).rglob("*"):
                if any(part.endswith(".egg-info") for part in path.parts):
                    continue
                if not path.is_file() or path.suffix in {".pyc", ".png", ".jpg", ".pdf"}:
                    continue
                if path.name == "config.py":
                    # The validator intentionally names forbidden path prefixes.
                    continue
                text = path.read_text(encoding="utf-8")
                for token in forbidden:
                    self.assertNotIn(token, text, f"{token!r} leaked into {path}")

    def test_code_repository_contains_no_model_binary(self):
        model_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}
        offenders = [
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and path.suffix in model_suffixes
            and not any(part.startswith(".venv") for part in path.parts)
            and ".git" not in path.parts
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
