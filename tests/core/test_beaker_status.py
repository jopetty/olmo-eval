"""Tests for BeakerStatusReporter."""

import unittest
from unittest import mock

from olmo_eval.common import beaker_status


class BeakerStatusReporterTest(unittest.TestCase):
    def test_disabled_when_env_unset(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            reporter = beaker_status.BeakerStatusReporter()
        self.assertFalse(reporter.enabled)
        reporter.update("hello")

    def test_throttles_updates_within_interval(self) -> None:
        env = {
            "BEAKER_WORKLOAD_ID": "wl_123",
            "GIT_COMMIT": "abc123",
            "GIT_BRANCH": "main",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            reporter = beaker_status.BeakerStatusReporter(min_interval=60.0)

            self.assertTrue(reporter.enabled)

            fake_client = mock.MagicMock()
            fake_workload = mock.MagicMock()
            reporter._client = fake_client
            reporter._workload = fake_workload

            with mock.patch("time.monotonic", side_effect=[0.0, 1.0, 61.0]):
                reporter.update("first")
                reporter.update("second")
                reporter.update("third")

        self.assertEqual(fake_client.workload.update.call_count, 2)
        suffix = "git_commit: abc123 git_branch: main"
        fake_client.workload.update.assert_any_call(fake_workload, description=f"first {suffix}")
        fake_client.workload.update.assert_any_call(fake_workload, description=f"third {suffix}")

    def test_git_suffix_uses_unknown_when_env_missing(self) -> None:
        env = {"BEAKER_WORKLOAD_ID": "wl_123"}
        with mock.patch.dict("os.environ", env, clear=True):
            reporter = beaker_status.BeakerStatusReporter(min_interval=0.0)

            fake_client = mock.MagicMock()
            fake_workload = mock.MagicMock()
            reporter._client = fake_client
            reporter._workload = fake_workload

            reporter.update("hello")

        fake_client.workload.update.assert_called_once_with(
            fake_workload, description="hello git_commit: unknown git_branch: unknown"
        )

    def test_force_bypasses_throttle(self) -> None:
        with mock.patch.dict("os.environ", {"BEAKER_WORKLOAD_ID": "wl_xyz"}, clear=True):
            reporter = beaker_status.BeakerStatusReporter(min_interval=60.0)

        fake_client = mock.MagicMock()
        reporter._client = fake_client
        reporter._workload = mock.MagicMock()

        with mock.patch("time.monotonic", side_effect=[0.0, 1.0]):
            reporter.update("a")
            reporter.update("b", force=True)

        self.assertEqual(fake_client.workload.update.call_count, 2)

    def test_dedupes_identical_messages(self) -> None:
        with mock.patch.dict("os.environ", {"BEAKER_WORKLOAD_ID": "wl_xyz"}, clear=True):
            reporter = beaker_status.BeakerStatusReporter(min_interval=0.0)

        fake_client = mock.MagicMock()
        reporter._client = fake_client
        reporter._workload = mock.MagicMock()

        with mock.patch("time.monotonic", side_effect=[0.0, 1.0, 2.0]):
            reporter.update("same")
            reporter.update("same")
            reporter.update("different")

        self.assertEqual(fake_client.workload.update.call_count, 2)


if __name__ == "__main__":
    unittest.main()
