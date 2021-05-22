import os
import time
from mpfls.tests.mpf_ls_base_test import MpfLsBaseTest, test_config_directory


class TestCompletion(MpfLsBaseTest):

    @test_config_directory("completion")
    def test_completion(self):
        self.maxDiff = None
        machine_config = self.get_test_file_path(["config", "config.yaml"])
        position_start = {"line": 0, "character": 0}
        self.assertEqual(
            {'isIncomplete': False, 'items': [
                {'label': '#config_version=5', 'kind': 1, 'detail': '', 'documentation': '',
                 'sortText': '#config_version=5', 'insertText': '#config_version=5\n'},
                {'label': '#show_version=5', 'kind': 1, 'detail': '', 'documentation': '',
                 'sortText': '#show_version=5', 'insertText': '#show_version=5\n'}
            ]},
            self.mpfls.completions(machine_config, position_start))

        position1 = {"line": 1, "character": 0}
        print(self.mpfls.completions(machine_config, position1))

        position2 = {"line": 10, "character": 4}
        position3 = {"line": 6, "character": 16}

        # config = a._load_config(filename)
        #
        # print(config)
        # print(config.lc.data)
        #
        # for item in config.values():
        #
        #     if hasattr(item, "lc"):
        #         print(item.lc.data)
        # print(config.lc.line)
        #
        # print(a._get_position_path(config, ))

        #print(a.workspace.get_root_document())
        #print(a.workspace.get_complete_config())


        #print(a._get_position_path(config, position1))
        #print(a.completions(filename, position1))
        t1 = time.time()
        print(self.mpfls.definitions(machine_config, position1))
        t2 = time.time()
        print(self.mpfls.definitions(machine_config, position1))
        t3 = time.time()

        print(t2 - t1, t3 - t2)
        # print(a._get_position_path(config, position2))
        # print(a.completions(filename, position2))
        # print(a._get_position_path(config, position3))
        # print(a.completions(filename, position3))
