"""Environment diagnostics fail on absent dependencies without installing anything."""

import importlib.metadata as metadata
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.check_environment import inspect_environment


class EnvironmentReports(unittest.TestCase):
    def test_missing_dependencies_fail_the_selected_profile(self):
        with patch("scripts.check_environment.metadata.version", side_effect=metadata.PackageNotFoundError), \
                patch("scripts.check_environment.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")):
            report = inspect_environment("policy")
        self.assertFalse(report["ok"])
        self.assertEqual(report["packages"], {})

    def test_dependency_conflicts_are_not_hidden_by_installed_versions(self):
        with patch("scripts.check_environment.metadata.version", return_value="1.0"), \
                patch("scripts.check_environment.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="incompatible dependency", stderr="")):
            report = inspect_environment("policy")
        self.assertFalse(report["ok"])
        self.assertEqual(report["pip_check"], "incompatible dependency")

    def test_source_profile_never_imports_or_checks_the_ml_stack(self):
        with patch("scripts.check_environment.metadata.version") as versions, \
                patch("scripts.check_environment.subprocess.run") as commands:
            report = inspect_environment("source")
        self.assertTrue(report["ok"])
        versions.assert_not_called()
        commands.assert_not_called()


if __name__ == "__main__":
    unittest.main()
