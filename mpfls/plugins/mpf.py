# Copyright 2018 Palantir Technologies, Inc.
import logging
from mpf.core.config_validator import ConfigValidator
from mpfls import hookimpl

from mpfls import lsp

log = logging.getLogger(__name__)

config_spec = None

@hookimpl
def pyls_initialize(config):
    global config_spec


@hookimpl
def pyls_completions(config, document, position):
    global config_spec
    # if not config_spec:
    #     return None

    key = "asd"
    value = "bcc"
    completions = [{
        'label': key,
        'kind': lsp.CompletionItemKind.Class,
        'detail': value,
        'documentation': value,
        'sortText': key,
        'insertText': value
    }]
    return completions



