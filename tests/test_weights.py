import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from madl.weights import verify_weight_manifest


class WeightManifestTests(unittest.TestCase):
    def test_manifest_verification_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "weight.bin"
            weight.write_bytes(b"madl")
            manifest = {
                "files": [
                    {
                        "path": "weight.bin",
                        "bytes": 4,
                        "sha256": hashlib.sha256(b"madl").hexdigest(),
                    }
                ]
            }
            (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify_weight_manifest(root), [])
            weight.write_bytes(b"changed")
            self.assertTrue(verify_weight_manifest(root))


if __name__ == "__main__":
    unittest.main()
