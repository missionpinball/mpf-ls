MPF Language Server
===================

Language server for MPF config.

Installation (from pypi)
------------------------

.. code-block:: bash

    pip3 install mpf-ls

Usage in IDE
------------

For any IntelliJ based IDE (such as PyCharm, WebStorm or PhpStorm) you need to
install a LSP (Language Server Protocol) plugin.
Then add ``mpfls`` as ``Raw command`` for ``yaml`` files in
"Settings -> Languages & Frameworks -> Language Server Protocol -> Server Definitions".

For vsCode install the extension in `vscode-client <https://github.com/missionpinball/mpf-ls/tree/master/vscode-client>`_.


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

