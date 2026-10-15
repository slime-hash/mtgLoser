# Project safety boundary

This project is strictly external and read-only with respect to the game.

- Allowed: ordinary screen pixels obtained from Windows Desktop Duplication,
  OBS, screenshots, or video files; local training/inference; a separate
  preview window; reading the public window title and client rectangle.
- Forbidden: opening the game process, reading/writing its memory, DLL or code
  injection, hooks inside the client, modifying game files, intercepting game
  traffic, sending keyboard/mouse/controller input, gameplay automation, or
  anti-cheat detection/evasion.
- Do not add a dependency capable of game input or process manipulation without
  stopping and obtaining a new explicit user decision.
- Do not claim that an external process is undetectable or officially allowed.
  Technical non-interaction is not a guarantee about publisher policy.
- Prefer offline analysis of user-created recordings whenever live capture is
  unnecessary.
