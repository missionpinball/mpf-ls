# Copyright 2017 Palantir Technologies, Inc.
import logging
import socketserver
import threading
from functools import partial

from mpf.core.config_validator import ConfigValidator
from mpf.file_interfaces.yaml_roundtrip import YamlRoundtrip
from pyls_jsonrpc.dispatchers import MethodDispatcher
from pyls_jsonrpc.endpoint import Endpoint
from pyls_jsonrpc.streams import JsonRpcStreamReader, JsonRpcStreamWriter

from . import lsp, _utils, uris
from .config import config
from .workspace import Workspace

log = logging.getLogger(__name__)


LINT_DEBOUNCE_S = 0.5  # 500 ms
PARENT_PROCESS_WATCH_INTERVAL = 10  # 10 s
MAX_WORKERS = 64
PYTHON_FILE_EXTENSIONS = ('.yaml')
CONFIG_FILEs = ('pycodestyle.cfg', 'setup.cfg', 'tox.ini', '.flake8')


class _StreamHandlerWrapper(socketserver.StreamRequestHandler, object):
    """A wrapper class that is used to construct a custom handler class."""

    delegate = None

    def setup(self):
        super(_StreamHandlerWrapper, self).setup()
        # pylint: disable=no-member
        self.delegate = self.DELEGATE_CLASS(self.rfile, self.wfile)

    def handle(self):
        self.delegate.start()
        # pylint: disable=no-member
        self.SHUTDOWN_CALL()


def start_tcp_lang_server(bind_addr, port, check_parent_process, handler_class):
    if not issubclass(handler_class, PythonLanguageServer):
        raise ValueError('Handler class must be an instance of PythonLanguageServer')

    def shutdown_server(*args):
        # pylint: disable=unused-argument
        log.debug('Shutting down server')
        # Shutdown call must be done on a thread, to prevent deadlocks
        stop_thread = threading.Thread(target=server.shutdown)
        stop_thread.start()

    # Construct a custom wrapper class around the user's handler_class
    wrapper_class = type(
        handler_class.__name__ + 'Handler',
        (_StreamHandlerWrapper,),
        {'DELEGATE_CLASS': partial(handler_class,
                                   check_parent_process=check_parent_process),
         'SHUTDOWN_CALL': shutdown_server}
    )

    server = socketserver.TCPServer((bind_addr, port), wrapper_class)
    server.allow_reuse_address = True

    try:
        log.info('Serving %s on (%s, %s)', handler_class.__name__, bind_addr, port)
        server.serve_forever()
    finally:
        log.info('Shutting down')
        server.server_close()


def start_io_lang_server(rfile, wfile, check_parent_process, handler_class):
    if not issubclass(handler_class, PythonLanguageServer):
        raise ValueError('Handler class must be an instance of PythonLanguageServer')
    log.info('Starting %s IO language server', handler_class.__name__)
    server = handler_class(rfile, wfile, check_parent_process)
    server.start()


class PythonLanguageServer(MethodDispatcher):
    """ Implementation of the Microsoft VSCode Language Server Protocol
    https://github.com/Microsoft/language-server-protocol/blob/master/versions/protocol-1-x.md
    """

    # pylint: disable=too-many-public-methods,redefined-builtin

    def __init__(self, rx, tx, check_parent_process=False):
        self.workspace = None
        self.config = None
        self.root_uri = None
        self.watching_thread = None
        self.workspaces = {}
        self.uri_workspace_mapper = {}

        self._jsonrpc_stream_reader = JsonRpcStreamReader(rx)
        self._jsonrpc_stream_writer = JsonRpcStreamWriter(tx)
        self._check_parent_process = check_parent_process
        self._endpoint = Endpoint(self, self._jsonrpc_stream_writer.write, max_workers=MAX_WORKERS)
        self._dispatchers = []
        self._shutdown = False

        validator = ConfigValidator(None, True, False)
        self.config_spec = validator.get_config_spec()

    def start(self):
        """Entry point for the server."""
        self._jsonrpc_stream_reader.listen(self._endpoint.consume)

    def __getitem__(self, item):
        """Override getitem to fallback through multiple dispatchers."""
        if self._shutdown and item != 'exit':
            # exit is the only allowed method during shutdown
            log.debug("Ignoring non-exit method during shutdown: %s", item)
            raise KeyError

        try:
            return super(PythonLanguageServer, self).__getitem__(item)
        except KeyError:
            # Fallback through extra dispatchers
            for dispatcher in self._dispatchers:
                try:
                    return dispatcher[item]
                except KeyError:
                    continue

        raise KeyError()

    def m_shutdown(self, **_kwargs):
        self._shutdown = True
        return None

    def m_exit(self, **_kwargs):
        self._endpoint.shutdown()
        self._jsonrpc_stream_reader.close()
        self._jsonrpc_stream_writer.close()

    def _match_uri_to_workspace(self, uri):
        workspace_uri = _utils.match_uri_to_workspace(uri, self.workspaces)
        return self.workspaces.get(workspace_uri, self.workspace)

    def _hook(self, hook_name, doc_uri=None, **kwargs):
        """Calls hook_name and returns a list of results from all registered handlers"""
        workspace = self._match_uri_to_workspace(doc_uri)
        doc = workspace.get_document(doc_uri) if doc_uri else None
        hook_handlers = self.config.plugin_manager.subset_hook_caller(hook_name, self.config.disabled_plugins)
        return hook_handlers(config=self.config, workspace=workspace, document=doc, **kwargs)

    def capabilities(self):
        server_capabilities = {
            'codeActionProvider': True,
            'codeLensProvider': {
                'resolveProvider': False,  # We may need to make this configurable
            },
            'completionProvider': {
                'resolveProvider': False,  # We know everything ahead of time
                'triggerCharacters': ['.']
            },
            'documentFormattingProvider': True,
            'documentHighlightProvider': True,
            'documentRangeFormattingProvider': True,
            'documentSymbolProvider': True,
            'definitionProvider': True,
            'executeCommandProvider': {
                'commands': flatten(self._hook('pyls_commands'))
            },
            'hoverProvider': True,
            'referencesProvider': True,
            'renameProvider': True,
            'signatureHelpProvider': {
                'triggerCharacters': ['(', ',', '=']
            },
            'textDocumentSync': {
                'change': lsp.TextDocumentSyncKind.INCREMENTAL,
                'save': {
                    'includeText': True,
                },
                'openClose': True,
            },
            'workspace': {
                'workspaceFolders': {
                    'supported': True,
                    'changeNotifications': True
                }
            },
            'experimental': merge(self._hook('pyls_experimental_capabilities'))
        }
        log.info('Server capabilities: %s', server_capabilities)
        return server_capabilities

    def m_initialize(self, processId=None, rootUri=None, rootPath=None, initializationOptions=None, **_kwargs):
        log.debug('Language server initialized with %s %s %s %s', processId, rootUri, rootPath, initializationOptions)
        if rootUri is None:
            rootUri = uris.from_fs_path(rootPath) if rootPath is not None else ''

        self.workspaces.pop(self.root_uri, None)
        self.root_uri = rootUri
        self.workspace = Workspace(rootUri, self._endpoint)
        self.workspaces[rootUri] = self.workspace
        self.config = config.Config(rootUri, initializationOptions or {},
                                    processId, _kwargs.get('capabilities', {}))
        self._dispatchers = self._hook('pyls_dispatchers')
        self._hook('pyls_initialize')

        if self._check_parent_process and processId is not None and self.watching_thread is None:
            def watch_parent_process(pid):
                # exit when the given pid is not alive
                if not _utils.is_process_alive(pid):
                    log.info("parent process %s is not alive", pid)
                    self.m_exit()
                else:
                    threading.Timer(PARENT_PROCESS_WATCH_INTERVAL, watch_parent_process, args=[pid]).start()

            self.watching_thread = threading.Thread(target=watch_parent_process, args=(processId,))
            self.watching_thread.daemon = True
            self.watching_thread.start()
        # Get our capabilities
        return {'capabilities': self.capabilities()}

    def m_initialized(self, **_kwargs):
        pass

    def code_actions(self, doc_uri, range, context):
        return flatten(self._hook('pyls_code_actions', doc_uri, range=range, context=context))

    def code_lens(self, doc_uri):
        return flatten(self._hook('pyls_code_lens', doc_uri))

    def _load_config(self, source):
        loader = YamlRoundtrip()
        try:
            config_dict = loader.process(source)
        except:
            return {}
        return config_dict

    def _get_position_path(self, config, position):
        line = position['line']
        character = position["character"]
        candidate_key = None
        if hasattr(config, "lc"):
            for key, lc in config.lc.data.items():
                if len(lc) >= 4 and ((lc[0] <= line and lc[3] <= character) or (lc[1] < character and lc[2] < line)):
                    candidate_key = key

        if candidate_key is not None:
            return [candidate_key] + self._get_position_path(config[candidate_key], position)
        else:
            return []

    def _get_settings_suggestion(self, settings_name):
        suggestions = []
        for key, value in self.config_spec.get(settings_name, {}).items():
            if key.startswith("__"):
                continue
            if value[1].startswith("subconfig") or value[0] in ("list", "dict"):
                insert_text = key + ":\n      "
            else:
                insert_text = key + ": "

            suggestions.append((key, insert_text, ""))

        return suggestions

    def _get_settings_value_suggestions(self, config, settings):
        if settings[1].startswith("enum"):
            values = settings[1][5:-1].split(",")
            suggestions = [(value, value + "\n", "") for value in values]
        elif settings[1].startswith("machine"):
            device = settings[1][8:-1]
            devices = config.get(device, {})
            suggestions = [(device, device + "\n    ", "") for device in devices]
        elif settings[1].startswith("subconfig"):
            settings_name = settings[1][10:-1]
            suggestions = self._get_settings_suggestion(settings_name)
        elif settings[1] == "bool":
            suggestions = [("True", "True\n    ", "(Default)" if "True" == settings[2] else ""),
                           ("False", "False\n    ", "(Default)" if "False" == settings[2] else "")]
        else:
            suggestions = []

        return suggestions

    def completions(self, doc_uri, position):
        completions = []

        if position["line"] == 0 and position["character"] == 0:
            return {
                'isIncomplete': False,
                'items': [{
                        'label': "#config_version=5",
                        'kind': lsp.CompletionItemKind.Text,
                        'detail': "",
                        'documentation': "",
                        'sortText': "#config_version=5",
                        'insertText': "#config_version=5\n"
                    },
                    {
                        'label': "#show_version=5",
                        'kind': lsp.CompletionItemKind.Text,
                        'detail': "",
                        'documentation': "",
                        'sortText': "#show_version=5",
                        'insertText': "#show_version=5\n"
                    }
                ]
            }

        document = self.workspace.get_document(doc_uri)
        config = self._load_config(document.source)
        path = self._get_position_path(config, position)

        if len(path) == 0:
            # global level -> all devices are valid
            # TODO: check if this is a mode or machine file
            suggestions = [(key, key + ":\n  ", "") for key, value in self.config_spec.items()
                           if "machine" in value.get("__valid_in__", [])]
        elif len(path) == 1:
            # device name level -> no suggestions
            suggestions = []
        elif len(path) == 2:
            # device level -> suggest config options
            suggestions = self._get_settings_suggestion(path[0])
        elif len(path) == 3:
            # settings level
            device_settings = self.config_spec.get(path[0], {})
            attribute_settings = device_settings.get(path[2], ["", "", ""])
            suggestions = self._get_settings_value_suggestions(config, attribute_settings)
        elif len(path) >= 3:
            device_settings = self.config_spec.get(path[0], {})
            for i in range(2, len(path) - 1):
                attribute_settings = device_settings.get(path[i], ["", "", ""])
                if attribute_settings[1].startswith("subconfig"):
                    settings_name = attribute_settings[1][10:-1]
                    device_settings = self.config_spec.get(settings_name, {})
                else:
                    return []

            attribute_settings = device_settings.get(path[len(path) - 1], ["", "", ""])
            suggestions = self._get_settings_value_suggestions(config, attribute_settings)
        else:
            suggestions = []

        for key, insertText, value in suggestions:
            completions.append(
                {
                    'label': key,
                    'kind': lsp.CompletionItemKind.Property,
                    'detail': "{}".format(value),
                    'documentation': "{} {}".format(key, value),
                    'sortText': key,
                    'insertText': insertText
                }
            )

        return {
            'isIncomplete': False,
            'items': completions
        }

    def definitions(self, doc_uri, position):
        return flatten(self._hook('pyls_definitions', doc_uri, position=position))

    def document_symbols(self, doc_uri):
        return flatten(self._hook('pyls_document_symbols', doc_uri))

    def execute_command(self, command, arguments):
        return self._hook('pyls_execute_command', command=command, arguments=arguments)

    def format_document(self, doc_uri):
        return self._hook('pyls_format_document', doc_uri)

    def format_range(self, doc_uri, range):
        return self._hook('pyls_format_range', doc_uri, range=range)

    def highlight(self, doc_uri, position):
        return flatten(self._hook('pyls_document_highlight', doc_uri, position=position)) or None

    def hover(self, doc_uri, position):
        return self._hook('pyls_hover', doc_uri, position=position) or {'contents': ''}

    @_utils.debounce(LINT_DEBOUNCE_S, keyed_by='doc_uri')
    def lint(self, doc_uri, is_saved):
        # Since we're debounced, the document may no longer be open
        workspace = self._match_uri_to_workspace(doc_uri)
        if doc_uri in workspace.documents:
            workspace.publish_diagnostics(
                doc_uri,
                flatten(self._hook('pyls_lint', doc_uri, is_saved=is_saved))
            )

    def references(self, doc_uri, position, exclude_declaration):
        return flatten(self._hook(
            'pyls_references', doc_uri, position=position,
            exclude_declaration=exclude_declaration
        ))

    def rename(self, doc_uri, position, new_name):
        return self._hook('pyls_rename', doc_uri, position=position, new_name=new_name)

    def signature_help(self, doc_uri, position):
        return self._hook('pyls_signature_help', doc_uri, position=position)

    def m_text_document__did_close(self, textDocument=None, **_kwargs):
        workspace = self._match_uri_to_workspace(textDocument['uri'])
        workspace.rm_document(textDocument['uri'])

    def m_text_document__did_open(self, textDocument=None, **_kwargs):
        workspace = self._match_uri_to_workspace(textDocument['uri'])
        workspace.put_document(textDocument['uri'], textDocument['text'], version=textDocument.get('version'))
        self._hook('pyls_document_did_open', textDocument['uri'])
        self.lint(textDocument['uri'], is_saved=True)

    def m_text_document__did_change(self, contentChanges=None, textDocument=None, **_kwargs):
        workspace = self._match_uri_to_workspace(textDocument['uri'])
        for change in contentChanges:
            workspace.update_document(
                textDocument['uri'],
                change,
                version=textDocument.get('version')
            )
        self.lint(textDocument['uri'], is_saved=False)

    def m_text_document__did_save(self, textDocument=None, **_kwargs):
        self.lint(textDocument['uri'], is_saved=True)

    def m_text_document__code_action(self, textDocument=None, range=None, context=None, **_kwargs):
        return self.code_actions(textDocument['uri'], range, context)

    def m_text_document__code_lens(self, textDocument=None, **_kwargs):
        return self.code_lens(textDocument['uri'])

    def m_text_document__completion(self, textDocument=None, position=None, **_kwargs):
        return self.completions(textDocument['uri'], position)

    def m_text_document__definition(self, textDocument=None, position=None, **_kwargs):
        return self.definitions(textDocument['uri'], position)

    def m_text_document__document_highlight(self, textDocument=None, position=None, **_kwargs):
        return self.highlight(textDocument['uri'], position)

    def m_text_document__hover(self, textDocument=None, position=None, **_kwargs):
        return self.hover(textDocument['uri'], position)

    def m_text_document__document_symbol(self, textDocument=None, **_kwargs):
        return self.document_symbols(textDocument['uri'])

    def m_text_document__formatting(self, textDocument=None, _options=None, **_kwargs):
        # For now we're ignoring formatting options.
        return self.format_document(textDocument['uri'])

    def m_text_document__rename(self, textDocument=None, position=None, newName=None, **_kwargs):
        return self.rename(textDocument['uri'], position, newName)

    def m_text_document__range_formatting(self, textDocument=None, range=None, _options=None, **_kwargs):
        # Again, we'll ignore formatting options for now.
        return self.format_range(textDocument['uri'], range)

    def m_text_document__references(self, textDocument=None, position=None, context=None, **_kwargs):
        exclude_declaration = not context['includeDeclaration']
        return self.references(textDocument['uri'], position, exclude_declaration)

    def m_text_document__signature_help(self, textDocument=None, position=None, **_kwargs):
        return self.signature_help(textDocument['uri'], position)

    def m_workspace__did_change_configuration(self, settings=None):
        self.config.update((settings or {}).get('pyls', {}))
        for workspace_uri in self.workspaces:
            workspace = self.workspaces[workspace_uri]
            for doc_uri in workspace.documents:
                self.lint(doc_uri, is_saved=False)

    def m_workspace__did_change_workspace_folders(self, added=None, removed=None, **_kwargs):
        for removed_info in removed:
            removed_uri = removed_info['uri']
            self.workspaces.pop(removed_uri)

        for added_info in added:
            added_uri = added_info['uri']
            self.workspaces[added_uri] = Workspace(added_uri, self._endpoint)

        # Migrate documents that are on the root workspace and have a better
        # match now
        doc_uris = list(self.workspace._docs.keys())
        for uri in doc_uris:
            doc = self.workspace._docs.pop(uri)
            new_workspace = self._match_uri_to_workspace(uri)
            new_workspace._docs[uri] = doc

    def m_workspace__did_change_watched_files(self, changes=None, **_kwargs):
        changed_py_files = set()
        config_changed = False
        for d in (changes or []):
            if d['uri'].endswith(PYTHON_FILE_EXTENSIONS):
                changed_py_files.add(d['uri'])
            elif d['uri'].endswith(CONFIG_FILEs):
                config_changed = True

        if config_changed:
            self.config.settings.cache_clear()
        elif not changed_py_files:
            # Only externally changed python files and lint configs may result in changed diagnostics.
            return

        for workspace_uri in self.workspaces:
            workspace = self.workspaces[workspace_uri]
            for doc_uri in workspace.documents:
                # Changes in doc_uri are already handled by m_text_document__did_save
                if doc_uri not in changed_py_files:
                    self.lint(doc_uri, is_saved=False)

    def m_workspace__execute_command(self, command=None, arguments=None):
        return self.execute_command(command, arguments)


def flatten(list_of_lists):
    return [item for lst in list_of_lists for item in lst]


def merge(list_of_dicts):
    return {k: v for dictionary in list_of_dicts for k, v in dictionary.items()}
