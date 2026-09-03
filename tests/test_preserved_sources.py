import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_original_configs_and_notebook_match_recorded_hashes():
    expected = json.loads((ROOT / "tests/preservation_manifest.json").read_text())
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
