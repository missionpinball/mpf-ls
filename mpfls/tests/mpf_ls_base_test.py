import os
import unittest
from unittest.mock import MagicMock

import mpfls
from mpfls import uris
from mpfls.mpf_ls import MPFLanguageServer


def test_config_directory(config_directory):

    """Decorator to overwrite config directory for one test."""

    def test_decorator(fn):
        """Decorate function."""
        fn.config_directory = config_directory
        return fn
    return test_decorator


class MpfLsBaseTest(unittest.TestCase):

    def get_test_config_dir(self, test_dir):
        """Get path for test."""
        return os.path.abspath(os.path.join(mpfls.__path__[0], "tests", "machines", test_dir))

    def get_test_file_path(self, file_parts):
        return uris.from_fs_path(os.path.join(self.test_root, *file_parts))

    def get_directory(self) -> str:
        directory = getattr(getattr(self, self._testMethodName), "config_directory", None)
        if not directory:
            raise AssertionError("Missing directory annotation for test {}".format(self._testMethodName))

        return directory

    def setUp(self) -> None:
        self.rx = MagicMock()
        self.tx = MagicMock()
        self.mpfls = MPFLanguageServer(self.rx, self.tx)
        self.test_root = self.get_test_config_dir(self.get_directory())
        self.mpfls.m_initialize(None, rootUri=self.test_root)
