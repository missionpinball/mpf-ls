from unittest.mock import MagicMock

from mpfls.tests.mpf_ls_base_test import MpfLsBaseTest, test_config_directory


class TestDiagnostics(MpfLsBaseTest):

    @test_config_directory("diagnostics")
    def test_diagnostics(self):
        self.maxDiff = None
        machine_config = self.get_test_file_path(["config", "config.yaml"])
        self.mpfls.workspace.publish_diagnostics = MagicMock()
        document = self.mpfls.workspace.get_document(machine_config)
        self.assertTrue(document)
        self.mpfls.lint_internal(machine_config)

        self.mpfls.workspace.publish_diagnostics.assert_called()
        call = self.mpfls.workspace.publish_diagnostics.mock_calls
        lints = call[0][1][1]
        print(lints)
