"""MPF Language Server."""
import logging
import os
import re
import socketserver
import threading
import traceback
from collections import namedtuple
from copy import deepcopy
from functools import partial

from mpf.core.config_processor import ConfigProcessor
from mpf.core.config_validator import ConfigValidator
from mpf.core.utility_functions import Util
from mpf.exceptions.config_file_error import ConfigFileError
from pyls_jsonrpc.dispatchers import MethodDispatcher
from pyls_jsonrpc.endpoint import Endpoint
from pyls_jsonrpc.streams import JsonRpcStreamReader, JsonRpcStreamWriter
from typing import List

from . import lsp, _utils, uris
from .config import config as lsp_configuration
from .workspace import Workspace, TYPE_MACHINE, TYPE_MODE, TYPE_SHOW

log = logging.getLogger(__name__)

EventInstance = namedtuple("EventInstance", ["event_name", "file_name", "config_section", "class_label",
                                             "desc", "args", "config_attribute", "device_name", "original_name"])

PostedEvent = namedtuple("PostedEvent", ["config_attribute", "event_name", "config_section", "device_name"])

EventHandler = namedtuple("EventHandler", ["event_name", "file_name", "config_section", "class_label",
                                           "desc", "args", "config_attribute", "device_name"])

LINT_DEBOUNCE_S = 0.5  # 500 ms
PARENT_PROCESS_WATCH_INTERVAL = 10  # 10 s
MAX_WORKERS = 64
MPF_FILE_EXTENSIONS = ('.yaml')
CONFIG_FILEs = ('mpfls.cfg')


class _StreamHandlerWrapper(socketserver.StreamRequestHandler, object):

    """A wrapper class that is used to construct a custom handler class."""

    delegate = None

    def setup(self):
        """Setup stream."""
        super(_StreamHandlerWrapper, self).setup()
        # pylint: disable=no-member
        self.delegate = self.DELEGATE_CLASS(self.rfile, self.wfile)

    def handle(self):
        """Handle stream."""
        self.delegate.start()
        # pylint: disable=no-member
        self.SHUTDOWN_CALL()


def start_tcp_lang_server(bind_addr, port, check_parent_process, handler_class):
    """Start TCP server for language server."""
    if not issubclass(handler_class, MPFLanguageServer):
        raise ValueError('Handler class must be an instance of MPFLanguageServer')

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
    """Start language server on stdin/stdout."""
    if not issubclass(handler_class, MPFLanguageServer):
        raise ValueError('Handler class must be an instance of MPFLanguageServer')
    log.info('Starting %s IO language server', handler_class.__name__)
    server = handler_class(rfile, wfile, check_parent_process)
    server.start()


class MPFLanguageServer(MethodDispatcher):

    """Implementation of the Microsoft VSCode Language Server Protocol.

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

        config_processor = ConfigProcessor(True, True)
        self.config_spec = config_processor.load_config_spec()
        self.validator = ConfigValidator(None, self.config_spec)

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
            return super(MPFLanguageServer, self).__getitem__(item)
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
                'commands': []
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
            'experimental': []
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
        self.config = lsp_configuration.Config(rootUri, initializationOptions or {},
                                               processId, _kwargs.get('capabilities', {}))

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
        log.warning("Code actions %s %s %s", doc_uri, range, context)
        return []

    def code_lens(self, doc_uri):
        log.warning("Code lens %s", doc_uri)
        return []

    def _get_start_of_token_at_position(self, lines, position):
        line = position['line']
        character = position["character"]

        try:
            current_line = lines[line]
        except IndexError:
            # line does not exist
            return position

        while character >= 0 and (character >= len(current_line) or
                                  current_line[character] not in (" ", ":", ",")):
            character -= 1

        return {"line": line, "character": character + 1}

    def _get_current_token(self, lines, start_position):
        line = start_position['line']
        start_character = character = start_position["character"]

        try:
            current_line = lines[line]
        except IndexError:
            # line does not exist
            return "", {"start": start_position, "end": start_position}

        while character < len(current_line) and current_line[character] not in (" ", ":", ",", "\n"):
            character += 1

        return current_line[start_character:character],\
               {"start": start_position, "end": {"line": line, "character": character}}

    def _get_position_path(self, config, position):
        line = position['line']
        character = position["character"]
        candidate_key = None
        token_range = [line, character, line, character]

        if hasattr(config, "lc") and config.lc.data:
            for key, lc in config.lc.data.items():
                if len(lc) == 4 and ((lc[0] <= line and lc[3] <= character) or (lc[1] < character and lc[2] < line)):
                    candidate_key = key
                    token_range = lc
                elif len(lc) == 2 and lc[0] <= line and lc[1] + 2 <= character:
                    candidate_key = key
                    token_range = lc

        if candidate_key is not None:
            path, child_range = self._get_position_path(config[candidate_key], position)
            if len(token_range) == 4:
                return [candidate_key] + path, child_range if child_range else token_range
            else:
                # skip lists in path but keep searching
                return path, child_range if child_range else token_range
        else:
            return [], None

    def _get_settings_suggestion(self, settings_name):
        suggestions = []
        spec = self._get_spec(settings_name)
        for key, value in spec.items():
            if key.startswith("__"):
                continue
            if not isinstance(value, list):
                insert_text = key + ":\n  "
            elif value[0] == "list" and value[1].startswith("subconfig"):
                insert_text = key + ":\n  - "
            elif value[0] == "dict" or value[1].startswith("subconfig"):
                insert_text = key + ":\n  "
            else:
                insert_text = key + ": "

            suggestions.append({
                    'label': key,
                    'kind': lsp.CompletionItemKind.Field,
                    'detail': "Setting {}".format(key),
                    'documentation': "Doc: Setting {}".format(key),
                    'sortText': key,
                    'insertText': insert_text
                })

        return suggestions

    def _find_device_in_config(self, document, device_type, device_name):
        found = []
        config = document.config_roundtrip
        device_config = config.get(device_type, {})
        if device_config:
            if device_name in device_config:
                lc = device_config.lc.data[device_name]
                range = {"start": {"line": lc[0], "character": lc[1]},
                         "end": {"line": lc[2], "character": lc[3]}}
                found.append({
                    "uri": uris.from_fs_path(document.path),
                    "range": range,
                })

        if 'config' in config:
            for file in Util.string_to_list(config['config']):
                path = os.path.join(os.path.split(document.path)[0], file)
                if os.path.exists(path):
                    sub_document = self.workspace.get_document(uris.from_fs_path(path))
                    found.extend(self._find_device_in_config(sub_document, device_type, device_name))

        return found

    def _get_definitions(self, device_type, device_name):
        # iterate all configs and find the device as high as possible in the hierarchy
        root_document = self.workspace.get_root_document()
        found = self._find_device_in_config(root_document, device_type, device_name)
        found.extend(self._find_device_in_config(self.workspace.get_mpf_config(), device_type, device_name))
        found.extend(self._find_device_in_config(self.workspace.get_mc_config(), device_type, device_name))

        config = self.workspace.get_complete_config()
        if "modes" in config:
            for mode in config['modes']:
                path = os.path.join(self.workspace.root_path, "modes", mode, "config", "{}.yaml".format(mode))
                if os.path.exists(path):
                    mode_document = self.workspace.get_document(uris.from_fs_path(path))
                    found.extend(self._find_device_in_config(mode_document, device_type, device_name))

        if device_type == "ball_devices":
            # special case for playfields
            found.extend(self._get_definitions("playfields", device_name))

        return found

    def _get_link_for_value(self, settings, device_name):
        if settings[1].startswith("machine"):
            device_type = settings[1][8:-1]
            found = self._get_definitions(device_type, device_name)
            return found
        elif settings[1] == "event_handler" or settings[0] == "event_handler":
            events = [event for event in self._get_known_events() if event.event_name == device_name]
            found = []
            for event in events:
                if event.config_section and event.device_name:
                    found.extend(self._get_definitions(event.config_section, event.device_name))
            return found
        elif settings[1] == "event_posted":
            # TODO: implement
            return None
            # events = self._get_known_event_handlers()
            # found = []
            # for event in events:
            #     if event.config_section and event.device_name:
            #         found.extend(self._get_definitions(event.config_section, event.device_name))
            # return found

        return None

    def _range_from_lc(self, document, lc):
        if len(lc) == 4:
            return {
                'start': {
                    'line': lc[0],
                    'character': lc[1]
                },
                'end': {
                    'line': lc[2],
                    'character': lc[3]
                }
            }
        else:
            return {
                'start': {
                    'line': lc[0],
                    'character': lc[1]
                },
                'end': {
                    'line': lc[0],
                    'character': len(document.lines[lc[0]])
                }
            }

    @staticmethod
    def _range_after_lc(document, lc):
        if len(lc) == 4:
            return {
                'start': {
                    'line': lc[2],
                    'character': lc[3]
                },
                'end': {
                    'line': lc[2],
                    'character': len(document.lines[lc[2]])
                }
            }
        else:
            return {
                'start': {
                    'line': lc[0],
                    'character': lc[1]
                },
                'end': {
                    'line': lc[0],
                    'character': len(document.lines[lc[0]])
                }
            }

    def _event_replace_placeholders(self, placeholders, event_name, event, device_name, device_config, config_section):
        event_instances = []
        for placeholder in placeholders:
            if placeholder == "name":
                event_name = event_name.replace("({})".format(placeholder), device_name)
            else:
                if placeholder not in self.config_spec[config_section]:
                    log.warning("Broken placeholder %s in event %s", placeholder, event)
                elif placeholder in device_config:
                    event_name = event_name.replace("({})".format(placeholder), device_config[placeholder])
                elif self.config_spec[config_section][placeholder][2]:
                    event_name = event_name.replace("({})".format(placeholder),
                                                    self.config_spec[config_section][placeholder][2])

        event_instances.append(EventInstance(event_name=event_name,
                                             file_name=event.file_name,
                                             config_section=event.config_section,
                                             class_label=event.class_label,
                                             desc=event.desc,
                                             args=event.args,
                                             config_attribute=event.config_attribute,
                                             original_name=event.event_name,
                                             device_name=device_name))
        return event_instances

    def _get_known_events(self) -> List[EventInstance]:
        """Return all known events."""
        known_events = []
        # get device event and merge them with our config
        device_events = self.workspace.get_device_events()
        config = self.workspace.get_complete_config()
        for event in device_events:
            if event.config_section:
                placeholders = re.findall(r'\(([^)]+)\)', event.event_name)
                config_sections = event.config_section if isinstance(event.config_section, list) else \
                    [event.config_section]
                for config_section in config_sections:
                    if config_section not in config:
                        continue
                    for device_name, device_config in config[config_section].items():
                        if event.config_attribute and event.config_attribute in device_config:
                            # ignore this as it is picked up later by parsing those events
                            pass
                        elif "tag" in placeholders:
                            placeholders_without_tag = list(placeholders)
                            placeholders_without_tag.remove("tag")
                            for tag in Util.string_to_list(device_config.get("tags", "")):
                                known_events.extend(self._event_replace_placeholders(
                                    placeholders_without_tag,
                                    event.event_name.replace("(tag)", tag),
                                    event, device_name, device_config, config_section))
                        else:
                            known_events.extend(self._event_replace_placeholders(placeholders, event.event_name,
                                                                                 event, device_name, device_config,
                                                                                 config_section))
            elif not event.config_section:
                known_events.append(EventInstance(event_name=event.event_name,
                                                  file_name=event.file_name,
                                                  config_section="",
                                                  class_label="",
                                                  desc=event.desc,
                                                  args=event.args,
                                                  config_attribute=event.config_attribute,
                                                  original_name=event.event_name,
                                                  device_name=""))

        # get posted events from our config
        posted_events = []
        for key, element_config in config.items():
            spec = self._get_spec(key)
            if spec.get("__type__", "") == "config":
                posted_events.extend(self._walk_config_for_event_handlers(spec, element_config, [key], ""))
            elif spec.get("__type__", "") == "list":
                # ignored here
                pass
            elif spec.get("__type__", "") == "config_player":
                # TODO: handle
                pass
            elif spec.get("__type__", "") == "device":
                for device_name, device_config in element_config.items():
                    posted_events.extend(self._walk_config_for_event_handlers(spec, device_config, key, device_name))

        for posted_event in posted_events:
            for event_ref in device_events:
                if event_ref.config_section == posted_event.config_section and \
                        event_ref.config_attribute == posted_event.config_attribute:
                    known_events.append(EventInstance(event_name=posted_event.event_name,
                                                      file_name=event_ref.file_name,
                                                      config_section=posted_event.config_section,
                                                      class_label=event_ref.class_label,
                                                      desc=event_ref.desc,
                                                      args=event_ref.args,
                                                      config_attribute=posted_event.config_attribute,
                                                      original_name=event_ref.event_name,
                                                      device_name=posted_event.device_name))
                    break
            else:
                known_events.append(EventInstance(event_name=posted_event.event_name,
                                                  file_name="",
                                                  config_section=posted_event.config_section,
                                                  class_label="",
                                                  desc="Posted event by {}".format(posted_event.config_section),
                                                  args={},
                                                  config_attribute=posted_event.config_attribute,
                                                  original_name=posted_event.event_name,
                                                  device_name=posted_event.device_name))

        return known_events

    @staticmethod
    def _walk_config_for_event_handlers(spec, config, device_type, device_name) -> List[PostedEvent]:
        events = []
        if not isinstance(config, dict):
            log.warning("INCORRECT CONFIG %s %s %s", device_type, device_name, config)
            return []
        for key, value in config.items():
            if key not in spec:
                continue
            if len(spec[key]) != 3:
                log.warning("WEIRD SPEC %s %s %s %s", device_type, device_name, key, spec[key])
                continue
            if spec[key][1] == "event_posted":
                if spec[key][0] == "list":
                    for event in Util.string_to_event_list(value):
                        events.append(PostedEvent(
                            config_attribute=key,
                            event_name=event,
                            config_section=device_type,
                            device_name=device_name))
                else:
                    events.append(PostedEvent(
                        config_attribute=key,
                        event_name=value,
                        config_section=device_type,
                        device_name=device_name))

        return events

    @staticmethod
    def _format_event_reference(event: EventInstance):
        desc = event.desc + "\n"
        for attribute, attribute_desc in event.args.items():
            desc += "\n{}: {}".format(attribute, attribute_desc)

        if event.class_label:
            desc += "\nPosted by device {}".format(event.class_label)

        desc += "\nDefined in: {}".format(event.file_name)

        return desc

    @staticmethod
    def _format_event_label(event: EventInstance):
        label = "Event: {}".format(event.original_name)
        if event.class_label:
            label += " from {}".format(event.class_label)
            if event.device_name:
                label += " " + event.device_name
        elif event.config_section:
            label += " from {}".format(event.config_section)

        return label

    def _get_settings_value_suggestions(self, settings):
        if settings[1].startswith("enum"):
            values = settings[1][5:-1].split(",")
            suggestions = [{
                    'label': value + (" (Default)" if value == settings[2] else ""),
                    'kind': lsp.CompletionItemKind.Value,
                    'detail': "Enum value {}".format(value),
                    'documentation': "TODO: Add docs",
                    'sortText': value,
                    'insertText': value
                } for value in values]
        elif settings[1].startswith("machine"):
            device = settings[1][8:-1]
            devices = self.workspace.get_complete_config().get(device, {})
            suggestions = [{
                    'label': device_name,
                    'kind': lsp.CompletionItemKind.Value,
                    'detail': "Reference to device {} of type {}".format(device_name, device),
                    'documentation': "TODO: Add docs",
                    'sortText': device_name,
                    'insertText': device_name
                } for device_name in devices]
        elif settings[1].startswith("subconfig"):
            settings_name = settings[1][10:-1]
            suggestions = self._get_settings_suggestion(settings_name)
        elif settings[1] == "event_handler" or settings[0] == "event_handler":
            events = self._get_known_events()
            suggestions = [
                {
                    'label': event.event_name,
                    'kind': lsp.CompletionItemKind.Value,
                    'detail': self._format_event_label(event),
                    'documentation': self._format_event_reference(event),
                    'sortText': event.event_name,
                    'insertText': event.event_name
                } for event in events]
        elif settings[1] == "bool":
            suggestions = [
                {
                    'label': "true" + (" (Default)" if "true" == settings[2].lower() else ""),
                    'kind': lsp.CompletionItemKind.Value,
                    'detail': "Boolean true",
                    'documentation': "This option is activated.",
                    'sortText': "true",
                    'insertText': "true"
                },
                {
                    'label': "false" + (" (Default)" if "false" == settings[2].lower() else ""),
                    'kind': lsp.CompletionItemKind.Value,
                    'detail': "Boolean false",
                    'documentation': "This option is deactivated.",
                    'sortText': "false",
                    'insertText': "false"
                },
            ]
        else:
            suggestions = []

        return suggestions

    def _walk_suggestion_path(self, device_settings, remaining_path):
        if len(remaining_path) > 1:
            attribute_settings = device_settings.get(remaining_path[0], ["", "", ""])
            if attribute_settings[0] == "dict" and attribute_settings[1].index(":subconfig("):
                settings_name = attribute_settings[1][attribute_settings[1].index("(") + 1:-1]
                device_settings = self._get_spec(settings_name)
                if len(remaining_path) > 2:
                    return self._walk_suggestion_path(device_settings, remaining_path[2:])
                else:
                    return self._get_settings_suggestion(settings_name)
            elif attribute_settings[1].startswith("subconfig"):
                settings_name = attribute_settings[1][10:-1]
                device_settings = self._get_spec(settings_name)
                return self._walk_suggestion_path(device_settings, remaining_path[1:])
            else:
                return []
        elif len(remaining_path) == 1:
            attribute_settings = device_settings.get(remaining_path[0], ["", "", ""])
            return self._get_settings_value_suggestions(attribute_settings)
        else:
            raise AssertionError("Path error")

    def _walk_suggestions(self, path):
        device_settings = self._get_spec(path[0])
        return self._walk_suggestion_path(device_settings, path[1:])

    def completions(self, doc_uri, position):
        completions = []

        if position["line"] == 0 and position["character"] == 0:
            return {
                'isIncomplete': False,
                'items': [{
                        'label': "#config_version=5",
                        'kind': lsp.CompletionItemKind.Keyword,
                        'detail': "",
                        'documentation': "",
                        'sortText': "#config_version=5",
                        'insertText': "#config_version=5\n"
                    },
                    {
                        'label': "#show_version=5",
                        'kind': lsp.CompletionItemKind.Keyword,
                        'detail': "",
                        'documentation': "",
                        'sortText': "#show_version=5",
                        'insertText': "#show_version=5\n"
                    }
                ]
            }

        document = self.workspace.get_document(doc_uri)
        token_start = self._get_start_of_token_at_position(document.lines, position)
        path, current_range = self._get_position_path(document.config_roundtrip, token_start)

        root_spec = self._get_spec(path[0]) if path else {}

        log.warning("Completions %s %s %s %s", doc_uri, position, path, root_spec)

        if len(path) == 0:
            # global level -> all devices are valid
            suggestions = [
                {
                    'label': key,
                    'kind': lsp.CompletionItemKind.Class,
                    'detail': "",
                    'documentation': "",
                    'sortText': key,
                    'insertText': key + ":\n  "
                } for key, value in self.config_spec.items()
                if document.config_type in value.get("__valid_in__", [])]
        elif len(path) == 1:
            if root_spec.get("__type__", "") in ("config", "list"):
                suggestions = self._get_settings_suggestion(path[0])
            else:
                # device name level -> no suggestions
                suggestions = []
        elif len(path) == 2:
            if root_spec.get("__type__", "") in ("config", "list"):
                # settings level
                suggestions = self._walk_suggestions(path)
            else:
                # device level -> suggest config options
                suggestions = self._get_settings_suggestion(path[0])
        elif len(path) >= 3:
            if root_spec.get("__type__", "") in ("config", "list"):
                # settings level
                suggestions = self._walk_suggestions(path)
            else:
                # device level -> suggest config options
                suggestions = self._walk_suggestions([path[0]] + path[2:])
        else:
            suggestions = []

        return {
            'isIncomplete': False,
            'items': suggestions
        }

    def _walk_definitions(self, path, token):
        device_settings = self._get_spec(path[0])
        # TODO: apply the same as for completion here
        for i in range(1, len(path) - 1):
            attribute_settings = device_settings.get(path[i], ["", "", ""])
            if attribute_settings[1].startswith("subconfig"):
                settings_name = attribute_settings[1][10:-1]
                device_settings = self._get_spec(settings_name)
            else:
                return []

        attribute_settings = device_settings.get(path[len(path) - 1], ["", "", ""])
        return self._get_link_for_value(attribute_settings, token)

    def definitions(self, doc_uri, position):
        log.warning("Definitions %s %s", doc_uri, position)

        document = self.workspace.get_document(doc_uri)
        token_start = self._get_start_of_token_at_position(document.lines, position)
        token, token_range = self._get_current_token(document.lines, token_start)
        path, current_range = self._get_position_path(document.config_roundtrip, token_start)
        root_spec = self._get_spec(path[0]) if path else {}

        log.warning("Definitions %s %s %s", token, token_start, token_range)

        if len(path) == 2:
            if root_spec.get("__type__", "") in ("config", "list"):
                return self._walk_definitions(path, token)
            else:
                return []
        elif len(path) >= 3:
            if root_spec.get("__type__", "") in ("config", "list"):
                return self._walk_definitions(path, token)
            else:
                return self._walk_definitions([path[0], path[2]], token)
        else:
            return []

    def _walk_devices(self, config):
        symbols = []
        for device_name, device_config in config.items():
            if hasattr(device_config, "lc"):
                for key, lc in device_config.lc.data.items():
                    range = {"start": {"line": lc[0], "character": lc[1]},
                               "end": {"line": lc[2], "character": lc[3]}}
                    symbols.append(
                        {
                            "name": key,
                            "detail": "",
                            "kind": lsp.SymbolKind.Class,
                            "deprecated": False,
                            "range": range,
                            "selectionRange": range,
                            "children": []
                        })
        return symbols

    def document_symbols(self, doc_uri):
        log.warning("Document symbols %s", doc_uri)

        return []

        document = self.workspace.get_document(doc_uri)
        symbols = self._walk_devices(document.config_roundtrip)

        return symbols

    def execute_command(self, command, arguments):
        log.warning("Execute command %s %s", command, arguments)
        return None

    def format_document(self, doc_uri):
        log.warning("Format Document %s", doc_uri)
        return None

    def format_range(self, doc_uri, range):
        log.warning("Format Range %s %s", doc_uri, range)
        return None

    def highlight(self, doc_uri, position):
        log.warning("Highlight %s %s", doc_uri, position)

        return None

        document = self.workspace.get_document(doc_uri)
        token_start = self._get_start_of_token_at_position(document.lines, position)
        # TODO: get current element instead of the one left of the element
        path, current_range = self._get_position_path(document.config_roundtrip, token_start)

        if current_range:
            return [

                {"kind": lsp.DocumentHighlightKind.Read,
                 "range": {"start": {"line": current_range[0], "character": current_range[1]},
                           "end": {"line": current_range[2], "character": current_range[3]}}
                 },

            ]
        else:
            return None

    def _layout_attribute_defaults(self, default_settings):
        if default_settings == "None":
            return "Defaults to empty."
        elif default_settings == "":
            return "Required attribute."
        else:
            return "Default: {}".format(default_settings)

    def _layout_attribute_settings(self, attribute_settings):
        if not attribute_settings or attribute_settings == ["", "", ""]:
            return ""
        if attribute_settings == "ignore":
            return "Not validated by your IDE."

        if attribute_settings[0] == "single":
            prefix = ""
        elif attribute_settings[0] == "list":
            if attribute_settings[1] == "event_handler":
                return "List of event handlers. {}".format(self._layout_attribute_defaults(attribute_settings[2]))
            elif attribute_settings[1] == "event_list":
                return "List of events posted. {}".format(self._layout_attribute_defaults(attribute_settings[2]))
            prefix = "List of "
        elif attribute_settings[0] == "dict":
            prefix = "Dictionary of "
        elif attribute_settings[0] == "event_handler":
            return "Dict or list of event handlers. Default: {}".format(attribute_settings[2])
        else:
            prefix = attribute_settings[0] + " "

        if attribute_settings[1].startswith("machine"):
            device_type = attribute_settings[1][8:-1]
            type = "device of type {}".format(device_type)
        else:
            type = attribute_settings[1]

        if not prefix:
            type = type.capitalize()

        return "{}{}. {}".format(prefix, type, self._layout_attribute_defaults(attribute_settings[2]))

    def _walk_hover(self, path, token):
        device_settings = self._get_spec(path[0])
        skip_next = False
        for i in range(1, len(path)):
            if skip_next:
                skip_next = False
                continue

            attribute_settings = device_settings.get(path[i], ["", "", ""])
            if isinstance(attribute_settings, List) and attribute_settings[1].startswith("subconfig"):
                settings_name = attribute_settings[1][10:-1]
                device_settings = self._get_spec(settings_name)
            elif isinstance(attribute_settings, List) and attribute_settings[0] == "dict" and "subconfig" in attribute_settings[1]:
                settings_name = attribute_settings[1][attribute_settings[1].index("(") + 1:-1]
                device_settings = self._get_spec(settings_name)
                skip_next = True
            else:
                return {'contents': ""}

        attribute_settings = device_settings.get(token, ["", "", ""])
        return {'contents': self._layout_attribute_settings(attribute_settings)}

    def hover(self, doc_uri, position):
        log.warning("Hover %s %s", doc_uri, position)

        document = self.workspace.get_document(doc_uri)
        token_start = self._get_start_of_token_at_position(document.lines, position)
        token, token_range = self._get_current_token(document.lines, token_start)
        path, current_range = self._get_position_path(document.config_roundtrip, token_start)

        root_spec = self._get_spec(path[0]) if path else {}

        if len(path) == 1:
            if root_spec.get("__type__", "") in ("config", "list"):
                return self._walk_hover([path[0]], token)
            else:
                return {'contents': ""}
        if len(path) >= 2:
            if root_spec.get("__type__", "") in ("config", "list"):
                return self._walk_hover(path, token)
            else:
                return self._walk_hover([path[0]] + path[2:], token)
        else:
            return {'contents': ""}

    def _walk_diagnostics_root(self, document, config):
        diagnostics = []
        lines = document.lines
        if document.config_type in (TYPE_MACHINE, TYPE_MODE):
            if not document.source.startswith("#config_version=5") and len(lines) > 1:
                diagnostics.append(
                    {
                        'source': 'mpf-ls',
                        'code': "1",
                        'range': {
                            'start': {
                                'line': 0,
                                'character': 0
                            },
                            'end': {
                                'line': 0,
                                'character': len(lines[0])
                            }
                        },
                        'message': "Config version is missing/wrong. Put #config_version=5 into the first line.",
                        'severity': lsp.DiagnosticSeverity.Error,
                    }
                )
        elif document.config_type == TYPE_SHOW:
            if not document.source.startswith("#show_version=5") and len(lines) > 1:
                diagnostics.append(
                    {
                        'source': 'mpf-ls',
                        'code': "2",
                        'range': {
                            'start': {
                                'line': 0,
                                'character': 0
                            },
                            'end': {
                                'line': 0,
                                'character': len(lines[0])
                            }
                        },
                        'message': "Config version is missing/wrong. Put #show_version=5 into the first line.",
                        'severity': lsp.DiagnosticSeverity.Error,
                    }
                )
            if hasattr(config, "lc"):
                for key, lc in config.lc.data.items():
                    if not isinstance(config[key], (list, dict)):
                        diagnostics.append(
                            {
                                'source': 'mpf-ls',
                                'code': "15",
                                'range': self._range_from_lc(document, lc),
                                'message': "Shows need to be lists.",
                                'severity': lsp.DiagnosticSeverity.Warning,
                            })
                    # TODO: this is not correct. or is it?
                    if "duration" not in config[key] and "time" not in config[key]:
                        diagnostics.append(
                            {
                                'source': 'mpf-ls',
                                'code': "16",
                                'range': self._range_from_lc(document, lc),
                                'message': "Show step needs duration or time.",
                                'severity': lsp.DiagnosticSeverity.Warning,
                            })

            return diagnostics

        if hasattr(config, "lc"):
            for key, lc in config.lc.data.items():
                if key not in self.config_spec:
                    diagnostics.append(
                        {
                            'source': 'mpf-ls',
                            'code': "3",
                            'range': self._range_from_lc(document, lc),
                            'message': "Unknown config key.",
                            'severity': lsp.DiagnosticSeverity.Warning,
                        }
                    )
                elif document.config_type not in self.config_spec[key].get("__valid_in__", []):
                    diagnostics.append(
                        {
                            'source': 'mpf-ls',
                            'code': "4",
                            'range': self._range_from_lc(document, lc),
                            'message': 'Config key "{}" not valid in {} config. It is only valid in: {}'.format(
                                key, document.config_type, self.config_spec[key].get("__valid_in__", [])),
                            'severity': lsp.DiagnosticSeverity.Warning,
                        }
                    )

            for key, lc in config.lc.data.items():
                root_spec = self._get_spec(key)
                if root_spec:
                    diagnostics.extend(self._walk_diagnostics_devices(document, config[key], root_spec, key, lc))

        return diagnostics

    def _walk_diagnostics_devices(self, document, config, root_spec, root_key, lc):
        diagnostics = []

        type = root_spec.get("__type__", "")

        if not type:
            diagnostics.append(
                {
                    'source': 'mpf-ls',
                    'code': "5",
                    'range': self._range_from_lc(document, lc),
                    'message': 'Config key "{}" has an invalid type.'.format(root_key),
                    'severity': lsp.DiagnosticSeverity.Warning,
                }
            )
        elif type == "list":
            if not isinstance(config, (list, str)):
                diagnostics.append(
                    {
                        'source': 'mpf-ls',
                        'code': "6",
                        'range': self._range_from_lc(document, lc),
                        'message': 'Expected a list.',
                        'severity': lsp.DiagnosticSeverity.Warning,
                    }
                )
        elif type == "config":
            diagnostics.extend(self._walk_diagnostics(document, config, [root_key], root_key, lc))
        elif type in ("device", "config_dict"):
            if not isinstance(config, dict) or not hasattr(config, "lc"):
                diagnostics.append(
                    {
                        'source': 'mpf-ls',
                        'code': "8",
                        'range': self._range_from_lc(document, lc),
                        'message': 'Expected a dictionary.',
                        'severity': lsp.DiagnosticSeverity.Warning,
                    }
                )
            else:
                for key, lc_child in config.lc.data.items():
                    diagnostics.extend(self._walk_diagnostics(document, config[key], [root_key], root_key, lc_child))
        elif type == "config_player":
            # TODO: implement
            pass
        elif type == "named_lists":
            # TODO: implement
            pass
        else:
            diagnostics.append(
                {
                    'source': 'mpf-ls',
                    'code': "99",
                    'range': self._range_from_lc(document, lc),
                    'message': 'Config key "{}" has type {} which is not implemented yet.'.format(root_key, type),
                    'severity': lsp.DiagnosticSeverity.Information,
                }
            )

        return diagnostics

    def _walk_diagnostics_dict(self, document, config, path, lc, key_spec):
        diagnostics = []
        if not isinstance(config, dict):
            diagnostics.append(
                {
                    'source': 'mpf-ls',
                    'code': "8",
                    'range': self._range_from_lc(document, lc),
                    'message': 'Expected a dictionary.',
                    'severity': lsp.DiagnosticSeverity.Warning,
                }
            )
        else:
            lines = document.lines
            dict_key_spec, dict_value_spec = key_spec[1].split(":", 1)
            if hasattr(config, "lc"):
                for dict_key, dict_key_lc in config.lc.data.items():
                    dict_value = config[dict_key]
                    dict_value_lc = [dict_key_lc[2], dict_key_lc[3], dict_key_lc[2],
                                     len(lines[dict_key_lc[2]])]
                    diagnostics.extend(self._walk_diagnostics_value(document, dict_key, path,
                                                                    dict_key_spec, dict_key_lc))
                    diagnostics.extend(self._walk_diagnostics_value(document, dict_value, path,
                                                                    dict_value_spec, dict_value_lc))
            else:
                dict_lc = self._range_from_lc(document, lc)
                for dict_key, dict_value in config.items():
                    diagnostics.extend(self._walk_diagnostics_value(document, dict_key, path,
                                                                    dict_key_spec, dict_lc))
                    diagnostics.extend(self._walk_diagnostics_value(document, dict_value, path,
                                                                    dict_value_spec, dict_lc))

        return diagnostics

    def _get_spec(self, spec_name_str):
        try:
            spec_names = spec_name_str.split(",")
        except Exception as e:
            raise AssertionError("Failed to split {}".format(spec_name_str)) from e

        spec = deepcopy(self.config_spec.get(spec_names[0], {}))
        for spec_name in spec_names[1:]:
            spec.update(self.config_spec.get(spec_name, {}))

        if spec.get("__type__") == "device":
            spec.update(self.config_spec["device"])
        if spec.get("__parent__", None):
            spec.update(self.config_spec.get(spec["__parent__"], {}))
        return spec

    def _walk_diagnostics(self, document, config, path, spec_name, lc):
        diagnostics = []
        lines = document.lines
        if not hasattr(config, "lc") or not config.lc.data:
            diagnostics.append(
                {
                    'source': 'mpf-ls',
                    'code': "999",
                    'range': self._range_from_lc(document, lc),
                    'message': 'Could not parse config.',
                    'severity': lsp.DiagnosticSeverity.Hint,
                }
            )
        elif not isinstance(config, dict):
            diagnostics.append(
                {
                    'source': 'mpf-ls',
                    'code': "9",
                    'range': self._range_from_lc(document, lc),
                    'message': 'Expected a dictionary. Got: {}'.format(config),
                    'severity': lsp.DiagnosticSeverity.Warning,
                }
            )
        else:
            spec = self._get_spec(spec_name)
            for key, child_lc in config.lc.data.items():
                if key in spec:
                    key_spec = spec[key]
                    if key_spec == "ignore":
                        continue
                    try:
                        if str(key_spec[2]) == str(config[key]):
                            diagnostics.append(
                                {
                                    'source': 'mpf-ls',
                                    'code': "10",
                                    'range': self._range_after_lc(document, child_lc),
                                    'message': 'Value is equal to default.',
                                    'severity': lsp.DiagnosticSeverity.Information,
                                }
                            )
                    except:
                        pass
                    if not isinstance(key_spec, List):
                        pass
                    elif key_spec[0] == "single":
                        if not hasattr(config[key], "lc"):
                            if config[key]:
                                value_lc = [child_lc[2], child_lc[3], child_lc[2], len(lines[child_lc[2]])]
                            else:
                                value_lc = child_lc
                        else:
                            value_lc = config[key].lc
                        diagnostics.extend(self._walk_diagnostics_value(document, config[key], path + [key],
                                                                        key_spec[1], value_lc))
                    elif key_spec[0] == "dict":
                        diagnostics.extend(self._walk_diagnostics_dict(document, config[key], path + [key], child_lc,
                                                                       key_spec))
                    elif key_spec[0] == "event_handler":
                        try:
                            event_config = Util.event_config_to_dict(config[key])
                        except Exception as e:
                            diagnostics.append(
                                {
                                    'source': 'mpf-ls',
                                    'code': "14",
                                    'range': self._range_after_lc(document, child_lc),
                                    'message': 'Could not convert event handler config.: {}'.format(e),
                                    'severity': lsp.DiagnosticSeverity.Error,
                                }
                            )
                        else:
                            diagnostics.extend(self._walk_diagnostics_dict(document, event_config, path + [key],
                                                                           child_lc, key_spec))
                    elif key_spec[0] in ("list", "event_list"):
                        try:
                            list = Util.string_to_list(config[key])
                        except Exception as e:
                            diagnostics.append(
                                {
                                    'source': 'mpf-ls',
                                    'code': "999",
                                    'range': self._range_after_lc(document, child_lc),
                                    'message': 'Expected a list: {}'.format(e),
                                    'severity': lsp.DiagnosticSeverity.Warning,
                                }
                            )
                        else:
                            if hasattr(config[key], "lc"):
                                for index, element_lc in config[key].lc.data.items():
                                    element = config[key][index]
                                    diagnostics.extend(self._walk_diagnostics_value(document, element, path + [key],
                                                                                    key_spec[1], element_lc))
                            else:
                                element_lc = [child_lc[0], child_lc[1], child_lc[2], len(lines[child_lc[2]])]
                                for element in list:
                                    diagnostics.extend(self._walk_diagnostics_value(document, element, path + [key],
                                                                                    key_spec[1], element_lc))

                    else:
                        diagnostics.append(
                            {
                                'source': 'mpf-ls',
                                'code': "12",
                                'range': self._range_from_lc(document, child_lc),
                                'message': 'Expected list "{}" in {}.'.format(key_spec[0], key, path),
                                'severity': lsp.DiagnosticSeverity.Information,
                            }
                        )

                elif "__allow_others__" not in spec:
                    diagnostics.append(
                        {
                            'source': 'mpf-ls',
                            'code': "10",
                            'range': self._range_from_lc(document, child_lc),
                            'message': 'Unknown config key "{}" in {}.'.format(key, path),
                            'severity': lsp.DiagnosticSeverity.Warning,
                        }
                    )
            for key, validator in spec.items():
                if isinstance(validator, List) and len(validator) == 3 and validator[2] == "" and key not in config:
                    diagnostics.append(
                        {
                            'source': 'mpf-ls',
                            'code': "11",
                            'range': self._range_from_lc(document, lc),
                            'message': 'Missing required key "{}".'.format(key),
                            'severity': lsp.DiagnosticSeverity.Warning,
                        }
                    )

        return diagnostics

    def _walk_diagnostics_value(self, document, config, path, value_type, value_lc):
        diagnostics = []
        if value_type.startswith("subconfig"):
            spec_name = value_type[10:-1]
            diagnostics.extend(self._walk_diagnostics(document, config, path, spec_name, value_lc))
        elif value_type.startswith("machine"):
            if config != "None":
                device_type = value_type[8:-1]
                found = self._get_definitions(device_type, config)
                if not found:
                    diagnostics.append(
                        {
                            'source': 'mpf-ls',
                            'code': "13",
                            'range': self._range_from_lc(document, value_lc),
                            'message': 'Could not find {} of type {}.'.format(config, device_type),
                            'severity': lsp.DiagnosticSeverity.Warning,
                        }
                    )
        elif value_type.startswith("template_"):
            # TODO: validate templates
            pass
        else:
            try:
                self.validator.validate_item(config, value_type, (("", ""), ""))
            except ConfigFileError as e:
                diagnostics.append(
                    {
                        'source': 'mpf-validator',
                        'code': e._error_no,
                        'range': self._range_from_lc(document, value_lc),
                        'message': str(e)+str(value_lc)+str(self._range_from_lc(document, value_lc)),
                        'severity': lsp.DiagnosticSeverity.Error,
                    }
                )
            except Exception as e:
                diagnostics.append(
                    {
                        'source': 'mpf-validator',
                        'code': "internal error",
                        'range': self._range_from_lc(document, value_lc),
                        'message': str(e),
                        'severity': lsp.DiagnosticSeverity.Error,
                    }
                )

        return diagnostics

    @_utils.debounce(LINT_DEBOUNCE_S, keyed_by='doc_uri')
    def lint(self, doc_uri, is_saved):
        # Since we're debounced, the document may no longer be open
        return self.lint_internal(doc_uri)

    def lint_internal(self, doc_uri):
        """Internal method without debounce."""
        workspace = self._match_uri_to_workspace(doc_uri)
        if doc_uri in workspace.documents:
            document = workspace.get_document(doc_uri)
            try:
                diagnostics = self._walk_diagnostics_root(document, document.config_roundtrip)
            except Exception as e:
                tb = traceback.format_exc()
                diagnostics = [
                    {
                        'source': 'mpf-ls',
                        'code': "998",
                        'range': {
                            'start': {
                                'line': 0,
                                'character': 0
                            },
                            'end': {
                                'line': 0,
                                'character': len(document.lines[0])
                            }
                        },
                        'message': "Internal error while verifying: {} {}".format(e, tb),
                        'severity': lsp.DiagnosticSeverity.Error,
                    }
                ]

            if document.parsing_failed:
                diagnostics.append(
                    {
                        'source': 'mpf-ls',
                        'code': "997",
                        'range': {
                            'start': {
                                'line': 0,
                                'character': 0
                            },
                            'end': {
                                'line': 0,
                                'character': len(document.lines[0])
                            }
                        },
                        'message': "Parsing of yaml failed.",
                        'severity': lsp.DiagnosticSeverity.Warning,
                    }
                )

            workspace.publish_diagnostics(
                doc_uri,
                diagnostics
            )

    def references(self, doc_uri, position, exclude_declaration):
        log.warning("References %s %s %s", doc_uri, position, exclude_declaration)
        return []

    def rename(self, doc_uri, position, new_name):
        log.warning("Rename %s %s %s", doc_uri, position, new_name)
        return None

    def signature_help(self, doc_uri, position):
        log.warning("Signature help %s %s", doc_uri, position)
        return None

    def m_text_document__did_close(self, textDocument=None, **_kwargs):
        workspace = self._match_uri_to_workspace(textDocument['uri'])
        workspace.rm_document(textDocument['uri'])

    def m_text_document__did_open(self, textDocument=None, **_kwargs):
        workspace = self._match_uri_to_workspace(textDocument['uri'])
        workspace.put_document(textDocument['uri'], textDocument['text'], version=textDocument.get('version'))
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
            if d['uri'].endswith(MPF_FILE_EXTENSIONS):
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
