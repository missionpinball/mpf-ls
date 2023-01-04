/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */
'use strict';

import { workspace, ExtensionContext, window } from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions, } from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
    const command = workspace.getConfiguration("mpfls").get<string>("executable");
	if(command) {
		const args = ["-vv"];
		const serverOptions: ServerOptions = {
			command,
			args,
		};
		const clientOptions: LanguageClientOptions = {
			documentSelector: ['yaml'],
			synchronize: {
				configurationSection: "mpfls"
			}
		}
		client = new LanguageClient("mpfls", "mpfls", serverOptions, clientOptions);
		client.start();
	} else {
		window.showErrorMessage('Lang Server Executable Config was not found!');
	}
}

export function deactivate(): Thenable<void> | undefined {
	if(!client) {
		return undefined;
	}
	return client.stop();
}