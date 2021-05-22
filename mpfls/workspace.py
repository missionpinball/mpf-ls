# Copyright 2017 Palantir Technologies, Inc.
import io
import logging
import pathlib

import mpfmc

import mpf
import os
import re

from mpf.core.utility_functions import Util
from mpf.file_interfaces.yaml_interface import YamlInterface
from mpf.file_interfaces.yaml_roundtrip import YamlRoundtrip
from mpf.parsers.event_reference_parser import EventReferenceParser, EventReference
from typing import List

from . import lsp, uris, _utils

log = logging.getLogger(__name__)

# TODO: this is not the best e.g. we capture numbers
RE_START_WORD = re.compile('[A-Za-z_0-9]*$')
RE_END_WORD = re.compile('^[A-Za-z_0-9]*')

TYPE_MACHINE = "machine"
TYPE_MODE = "mode"
TYPE_SHOW = "show"


class Workspace(object):

    M_PUBLISH_DIAGNOSTICS = 'textDocument/publishDiagnostics'
    M_APPLY_EDIT = 'workspace/applyEdit'
    M_SHOW_MESSAGE = 'window/showMessage'

    def __init__(self, root_uri, endpoint):
        self._root_uri = root_uri
        self._endpoint = endpoint
        self._root_uri_scheme = uris.urlparse(self._root_uri)[0]
        self._root_path = uris.to_fs_path(self._root_uri)
        self._docs = {}
        self._cached_config = {}
        self.mpf_path = str(pathlib.Path(mpf.__file__).parent.absolute())
        self.mc_path = str(pathlib.Path(mpfmc.__file__).parent.absolute())
        self.config_path = os.path.join(self._root_path, "config")
        self.mode_path = os.path.join(self._root_path, "modes")
        self.show_path = os.path.join(self._root_path, "shows")
        self._device_events = None

    def get_root_document(self):
        return self.get_document(uris.from_fs_path(os.path.join(self.config_path, "config.yaml")))

    def get_mpf_config(self):
        return self.get_document(uris.from_fs_path(os.path.join(self.mpf_path, "mpfconfig.yaml")))

    def get_mc_config(self):
        return self.get_document(uris.from_fs_path(os.path.join(self.mc_path, "mcconfig.yaml")))

    def get_device_events(self) -> List[EventReference]:
        if self._device_events is not None:
           return self._device_events

        event_parser = EventReferenceParser()
        self._device_events = event_parser.get_events_from_path([self.mpf_path, self.mc_path, self._root_path])
        return self._device_events

    def get_complete_config(self):
        if self._cached_config:
            return self._cached_config

        root_document = self.get_root_document()
        config = self._load_document_and_subconfigs(self.get_mpf_config())
        config = Util.dict_merge(config, self._load_document_and_subconfigs(root_document))
        if "modes" in config:
            for mode in config['modes']:
                path = os.path.join(self.mode_path, mode, "config", "{}.yaml".format(mode))
                if os.path.exists(path):
                    mode_document = self.get_document(uris.from_fs_path(path))
                    mode_config = self._load_document_and_subconfigs(mode_document)
                    mode_config.pop("mode", None)
                    config = Util.dict_merge(config, mode_config)

        self._cached_config = config

        return config

    def _load_document_and_subconfigs(self, root_document):
        config = root_document.config_simple
        if 'config' in config:
            for file in Util.string_to_list(config['config']):
                path = os.path.join(os.path.split(root_document.path)[0], file)
                if os.path.exists(path):
                    sub_document = self.get_document(uris.from_fs_path(path))
                    sub_config = self._load_document_and_subconfigs(sub_document)
                    config = Util.dict_merge(config, sub_config)

        return config

    @property
    def documents(self):
        return self._docs

    @property
    def root_path(self):
        return self._root_path

    @property
    def root_uri(self):
        return self._root_uri

    def is_local(self):
        return (self._root_uri_scheme == '' or self._root_uri_scheme == 'file') and os.path.exists(self._root_path)

    def get_document(self, doc_uri):
        """Return a managed document if-present, else create one pointing at disk.

        See https://github.com/Microsoft/language-server-protocol/issues/177
        """
        doc = self._docs.get(doc_uri)
        if doc:
            return doc

        doc = self._docs[doc_uri] = self._create_document(doc_uri)
        return doc

    def put_document(self, doc_uri, source, version=None):
        self._docs[doc_uri] = self._create_document(doc_uri, source=source, version=version)

    def rm_document(self, doc_uri):
        self._docs.pop(doc_uri)

    def update_document(self, doc_uri, change, version=None):
        self._cached_config = {}
        self._docs[doc_uri].apply_change(change)
        self._docs[doc_uri].version = version

    def apply_edit(self, edit):
        return self._endpoint.request(self.M_APPLY_EDIT, {'edit': edit})

    def publish_diagnostics(self, doc_uri, diagnostics):
        self._endpoint.notify(self.M_PUBLISH_DIAGNOSTICS, params={'uri': doc_uri, 'diagnostics': diagnostics})

    def show_message(self, message, msg_type=lsp.MessageType.Info):
        self._endpoint.notify(self.M_SHOW_MESSAGE, params={'type': msg_type, 'message': message})

    def source_roots(self, document_path):
        """Return the source roots for the given document."""
        files = _utils.find_parents(self._root_path, document_path, ['config', 'modes']) or []
        return list(set((os.path.dirname(project_file) for project_file in files))) or [self._root_path]

    def _create_document(self, doc_uri, source=None, version=None):
        path = uris.to_fs_path(doc_uri)

        if not path.startswith(os.path.abspath(self.root_path) + os.sep) and \
                not path.startswith(self.mpf_path + os.sep) and \
                not path.startswith(self.mc_path + os.sep):
            self.show_message("{} is not in workspace {}. MPF Language Server will not work.".format(path,
                                                                                                     self.root_path))

        if path.startswith(self.mode_path + os.sep):
            config_type = TYPE_MODE
        elif path.startswith(self.show_path + os.sep):
            config_type = TYPE_SHOW
        else:
            config_type = TYPE_MACHINE

        return Document(
            doc_uri, source=source, version=version, config_type=config_type
        )


class Document(object):

    def __init__(self, uri, config_type, source=None, version=None, local=True, extra_sys_path=None):
        self.uri = uri
        self.version = version
        self.path = uris.to_fs_path(uri)
        self.filename = os.path.basename(self.path)

        self._local = local
        self._source = source
        self._extra_sys_path = extra_sys_path or []
        self._config_simple = {}
        self._config_roundtrip = {}
        self._last_config_simple = {}
        self._last_config_roundtrip = {}
        self._loader_roundtrip = YamlRoundtrip()
        self._loader_simple = YamlInterface()
        self.config_type = config_type

    def invalidate_config(self):
        self._last_config_simple = {}
        self._config_roundtrip = {}

    @property
    def parsing_failed(self):
        return self._parsing_failed

    @property
    def config_roundtrip(self):
        if not self._config_roundtrip:
            self._load_config_roundtrip()

        if not self._config_roundtrip:
            if not self._last_config_roundtrip:
                return {}
            return self._last_config_roundtrip
        else:
            return self._config_roundtrip

    def _load_config_roundtrip(self):
        try:
            self._config_roundtrip = self._loader_roundtrip.process(self.source)
        except:
            self._parsing_failed = True
        else:
            self._parsing_failed = False
            self._last_config_roundtrip = self._config_roundtrip

    @property
    def config_simple(self):
        if not self._config_simple:
            self._load_config_simple()

        if not self._config_simple:
            if not self._last_config_simple:
                return {}
            return self._last_config_simple
        else:
            return self._config_simple

    def _load_config_simple(self):
        try:
            self._config_simple = self._loader_simple.process(self.source)
        except:
            self._parsing_failed = True
        else:
            self._parsing_failed = False
            self._last_config_simple = self._config_simple

    def __str__(self):
        return str(self.uri)

    @property
    def lines(self):
        return self.source.splitlines(True)

    @property
    def source(self):
        if self._source is None:
            with io.open(self.path, 'r', encoding='utf-8') as f:
                return f.read()
        return self._source

    def apply_change(self, change):
        """Apply a change to the document."""
        text = change['text']
        change_range = change.get('range')

        if not change_range:
            # The whole file has changed
            self._source = text
            self.invalidate_config()
            return

        start_line = change_range['start']['line']
        start_col = change_range['start']['character']
        end_line = change_range['end']['line']
        end_col = change_range['end']['character']

        # Check for an edit occuring at the very end of the file
        if start_line == len(self.lines):
            self._source = self.source + text
            self.invalidate_config()
            return

        new = io.StringIO()

        # Iterate over the existing document until we hit the edit range,
        # at which point we write the new text, then loop until we hit
        # the end of the range and continue writing.
        for i, line in enumerate(self.lines):
            if i < start_line:
                new.write(line)
                continue

            if i > end_line:
                new.write(line)
                continue

            if i == start_line:
                new.write(line[:start_col])
                new.write(text)

            if i == end_line:
                new.write(line[end_col:])

        self._source = new.getvalue()
        self.invalidate_config()

    def offset_at_position(self, position):
        """Return the byte-offset pointed at by the given position."""
        return position['character'] + len(''.join(self.lines[:position['line']]))

    def word_at_position(self, position):
        """Get the word under the cursor returning the start and end positions."""
        if position['line'] >= len(self.lines):
            return ''

        line = self.lines[position['line']]
        i = position['character']
        # Split word in two
        start = line[:i]
        end = line[i:]

        # Take end of start and start of end to find word
        # These are guaranteed to match, even if they match the empty string
        m_start = RE_START_WORD.findall(start)
        m_end = RE_END_WORD.findall(end)

        return m_start[0] + m_end[-1]

    def sys_path(self):
        # Copy our extra sys path
        path = list(self._extra_sys_path)

        return path
