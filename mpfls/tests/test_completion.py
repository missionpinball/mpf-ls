import os
import time
import unittest
from unittest.mock import MagicMock

import mpfls
from mpfls import uris
from mpfls.mpf_ls import MPFLanguageServer


class TestCompletion(unittest.TestCase):

    def get_test_config_dir(self, test_dir):
        """Get path for test."""
        return os.path.abspath(os.path.join(mpfls.__path__[0], "tests", "machines", test_dir))

    def get_test_file_path(self, file_parts):
        return uris.from_fs_path(os.path.join(self.test_root, *file_parts))

    def setUp(self) -> None:
        self.rx = MagicMock()
        self.tx = MagicMock()
        self.mpfls = MPFLanguageServer(self.rx, self.tx)
        self.test_root = self.get_test_config_dir("completion")
        self.mpfls.m_initialize(None, rootUri=self.test_root)

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
