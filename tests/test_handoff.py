"""HANDOFF.md carries the project state between sessions and between AIs. If it disappears, this test fails loudly."""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HandoffFile(unittest.TestCase):
    def test_handoff_exists_and_is_maintained(self):
        path = os.path.join(ROOT, "HANDOFF.md")
        self.assertTrue(os.path.exists(path), "HANDOFF.md is missing: restore it (it is the project's memory) and do not delete it")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for marker in ("DO NOT DELETE", "## START HERE", "## 5. Stage log", "## 7. Milestone plan", "## 12. Next action"):
            self.assertIn(marker, text, "HANDOFF.md lost its section: " + marker)
        self.assertGreater(len(text), 5000)


if __name__ == "__main__":
    unittest.main()
