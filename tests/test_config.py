import unittest
from pathlib import Path

from madl.config import MADLConfig


class ConfigTests(unittest.TestCase):
    def test_default_config_contains_only_portable_weight_references(self):
        config = MADLConfig()
        config.validate_portable()
        for value in config.weight_references().values():
            self.assertFalse(Path(value).is_absolute())
            self.assertNotIn("/mnt/", value)
            self.assertNotIn("/root/", value)

    def test_environment_can_override_weight_root(self):
        config = MADLConfig(weight_root="models")
        resolved = config.resolve_weight("agent_b")
        self.assertEqual(
            resolved.as_posix(),
            "models/agent_b_dualstream/madl_agent_b_dualstream_v1.pt",
        )


if __name__ == "__main__":
    unittest.main()
