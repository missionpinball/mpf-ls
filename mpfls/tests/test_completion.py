from unittest.mock import ANY

from mpfls.tests.mpf_ls_base_test import MpfLsBaseTest, test_config_directory


class TestCompletion(MpfLsBaseTest):

    @test_config_directory("completion")
    def test_completion(self):
        self.maxDiff = None
        machine_config = self.get_test_file_path(["config", "config.yaml"])
        position_start = {"line": 0, "character": 0}
        self.assertEqual(
            {'isIncomplete': False, 'items': [
                {'label': '#config_version=5', 'kind': 14, 'detail': '', 'documentation': '',
                 'sortText': '#config_version=5', 'insertText': '#config_version=5\n'},
                {'label': '#show_version=5', 'kind': 14, 'detail': '', 'documentation': '',
                 'sortText': '#show_version=5', 'insertText': '#show_version=5\n'}
            ]},
            self.mpfls.completions(machine_config, position_start))

        position_root = {"line": 1, "character": 0}
        completions_root = self.mpfls.completions(machine_config, position_root)

        self.assertIn(
            {'label': 'state_machines', 'kind': 7, 'detail': '', 'documentation': '',
             'sortText': 'state_machines', 'insertText': 'state_machines:\n  '},
            completions_root['items'])

        position_inside_state_machine_device = {"line": 3, "character": 4}
        completions_inside_state_machine_device = self.mpfls.completions(machine_config,
                                                                         position_inside_state_machine_device)

        self.assertIn(
            {'label': 'states', 'kind': 5, 'detail': 'Setting states', 'documentation': 'Doc: Setting states',
             'sortText': 'states', 'insertText': 'states:\n  '},
            completions_inside_state_machine_device['items'])

        self.assertIn(
            {'label': 'transitions', 'kind': 5, 'detail': 'Setting transitions',
             'documentation': 'Doc: Setting transitions', 'sortText': 'transitions',
             'insertText': 'transitions:\n  - '},
            completions_inside_state_machine_device['items'])

        position_inside_state_machine_transitions = {"line": 4, "character": 8}
        completions_inside_state_machine_transitions = self.mpfls.completions(machine_config,
                                                                              position_inside_state_machine_transitions)

        self.assertIn(
            {'label': 'source', 'kind': 5, 'detail': 'Setting source', 'documentation': 'Doc: Setting source',
             'sortText': 'source', 'insertText': 'source: '},
            completions_inside_state_machine_transitions['items'])
        self.assertIn(
            {'label': 'events', 'kind': 5, 'detail': 'Setting events', 'documentation': 'Doc: Setting events',
             'sortText': 'events', 'insertText': 'events: '},
            completions_inside_state_machine_transitions['items'])

        position_inside_state_machine_transitions_after_list = {"line": 10, "character": 8}
        completions_inside_state_machine_transitions_after_list = self.mpfls.completions(
            machine_config, position_inside_state_machine_transitions_after_list)

        self.assertIn(
            {'label': 'source', 'kind': 5, 'detail': 'Setting source', 'documentation': 'Doc: Setting source',
             'sortText': 'source', 'insertText': 'source: '},
            completions_inside_state_machine_transitions_after_list['items'])
        self.assertIn(
            {'label': 'events', 'kind': 5, 'detail': 'Setting events', 'documentation': 'Doc: Setting events',
             'sortText': 'events', 'insertText': 'events: '},
            completions_inside_state_machine_transitions_after_list['items'])

        position_inside_state_machine_transitions_events = {"line": 4, "character": 16}
        position_inside_state_machine_transitions_events2 = {"line": 8, "character": 10}
        completions_inside_state_machine_transitions_events = self.mpfls.completions(
            machine_config, position_inside_state_machine_transitions_events)
        completions_inside_state_machine_transitions_events2 = self.mpfls.completions(
            machine_config, position_inside_state_machine_transitions_events2)

        self.assertIn(
            {'label': 'ball_starting', 'kind': 12, 'detail': 'Event: ball_starting',
             'documentation': ANY, 'sortText': 'ball_starting',
             'insertText': 'ball_starting'},
            completions_inside_state_machine_transitions_events['items'])
        self.assertIn(
            {'label': 'ball_starting', 'kind': 12, 'detail': 'Event: ball_starting',
             'documentation': ANY, 'sortText': 'ball_starting',
             'insertText': 'ball_starting'},
            completions_inside_state_machine_transitions_events2['items'])
        self.assertIn(
            {'label': 'logicblock_test_counter_complete', 'kind': 12,
             'detail': "Event: logicblock_(name)_complete from ['counters', 'accruals', 'sequences']",
             'documentation': ANY, 'sortText': 'logicblock_test_counter_complete',
             'insertText': 'logicblock_test_counter_complete'},
            completions_inside_state_machine_transitions_events2['items'])

        position_inside_state_machine_states = {"line": 15, "character": 8}
        completions_inside_state_machine_states = self.mpfls.completions(machine_config,
                                                                         position_inside_state_machine_states)

        print(completions_inside_state_machine_states)
        self.assertIn(
            {'label': 'label', 'kind': 5, 'detail': 'Setting label', 'documentation': 'Doc: Setting label',
             'sortText': 'label', 'insertText': 'label: '},
            completions_inside_state_machine_states['items'])
