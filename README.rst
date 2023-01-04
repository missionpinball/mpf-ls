MPF Language Server
===================

Language server for MPF config.

Installation (from pypi)
------------------------

To use the MPF language server, you will have needed to install mpf and mpf-mc using the pipx commands from the install guide.

The command below will inject the mpf-language-server into your mpf virtualenv and make the mpfls command available.

.. code-block:: bash
  
    pipx inject mpf mpf-language-server --pip-args="--pre" --verbose --include-deps --include-apps

Usage in IDE
------------

Any IDE supported by language service will work. Here are a few examples:


IntelliJ Based IDE
~~~~~~~~~~~~~~~~~~

For any IntelliJ based IDE (such as PyCharm, WebStorm or PhpStorm) you need to
install a LSP (Language Server Protocol) plugin.
Then add ``mpfls`` as ``Raw command`` for ``yaml`` files in
"Settings -> Languages & Frameworks -> Language Server Protocol -> Server Definitions".

If you run mpf in Python Virtual Environment (venv) then mpfls also has to be run in that venv.
For example create a script that activate venv and run mpfls and add that script as ``Executable`` for ``yaml`` in
"Settings -> Languages & Frameworks -> Language Server Protocol -> Server Definitions".

VSCode
~~~~~~

For vsCode install the extension found here: `vscode-client <vscode-client>`_.  You will be installing the latest .vsix file from that location.

Emacs
~~~~~

Integration with Emacs is accomplished using `lsp-mode <https://github.com/emacs-lsp/lsp-mode>`_.

A minimal completion setup can be achieved with the :code:`lsp-mode`, :code:`yaml-mode`, :code:`company`, and :code:`lsp-company` packages.  Company is a general purpose completion package for Emacs.  :code:`lsp-company` is a helper package for using Company with :code`lsp-mode`.

1. Install :code:`lsp-mode`, :code:`company`, :code:`yaml-mode`, and :code:`lsp-company` by running :code:`M-x package-install` and following the instructions.
2. Add the following to your Emacs init file: ::

     ;; Register the mpfls server
     (require 'lsp-mode)
     (add-hook 'yaml-mode-hook #'lsp)
     (defvar lsp-language-id-configuration
       '((yaml-mode . "mpfls")))

     (lsp-register-client
       (make-lsp-client :new-connection (lsp-stdio-connection "mpfls")
         :major-modes '(yaml-mode)
         :server-id 'mpfls))
     (add-hook 'after-init-hook 'global-company-mode)

     ;; Configure company-lsp
     (require 'company-lsp)
     (push 'company-lsp company-backends)

Installation (from git for local development)
---------------------------------------------

If you want to contribute to this repository:

.. code-block:: bash

    git checkout https://github.com/missionpinball/mpf-ls/
    cd mpf-ls
    pip3 install -e .


License
-------

This project is made available under the MIT License.
Code is based on the `Python language server <https://github.com/palantir/python-language-server/>`_ (also MIT).
