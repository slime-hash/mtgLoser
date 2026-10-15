from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[1] / 'mcoc_vision'


def test_runtime_has_no_process_injection_or_synthetic_input_apis():
    source = '\n'.join(path.read_text(encoding='utf-8').casefold()
                       for path in SOURCE_ROOT.glob('*.py'))
    forbidden = {
        'openprocess', 'readprocessmemory', 'writeprocessmemory',
        'virtualallocex', 'createremotethread', 'setwindowshookex',
        'sendinput', 'keybd_event', 'mouse_event',
        'pyautogui', 'pydirectinput', 'pynput', 'win32process',
    }
    found = sorted(token for token in forbidden if token in source)
    assert not found, f'Forbidden game-interaction APIs found: {found}'


def test_runtime_does_not_open_network_or_subprocess_channels():
    source = '\n'.join(path.read_text(encoding='utf-8').casefold()
                       for path in SOURCE_ROOT.glob('*.py'))
    forbidden_imports = {
        'import socket', 'import requests', 'import urllib',
        'import subprocess', 'from socket', 'from requests', 'from urllib',
        'from subprocess',
    }
    found = sorted(token for token in forbidden_imports if token in source)
    assert not found, f'Forbidden runtime channel imports found: {found}'
