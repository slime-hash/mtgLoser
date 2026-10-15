# Safety boundary

MCOC Vision observes pixels; it does not interact with the game client.

## Permitted architecture

- Capture completed desktop frames through Windows Desktop Duplication (DXGI).
- Capture an OBS virtual camera or read a user-provided video/image.
- Locate a visible window and read its public screen rectangle through Win32 UI
  functions so the desktop capture can be cropped.
- Run detection, tracking, annotation, training, and reporting locally.
- Display results in a separate OpenCV window.

The direct-capture implementation does **not** obtain a process handle for the
game. DXcam duplicates the monitor output, and the crop is calculated from the
window's public client rectangle.

## Prohibited architecture

- Process memory access or inspection.
- DLL injection, client hooks, or in-process overlays.
- Modification of game binaries, assets, configuration, or anti-cheat state.
- Network interception or communication with game services.
- Synthetic keyboard, mouse, touch, or controller input.
- Automated gameplay, macros, bots, or real-time action execution.
- Hiding the tool from anti-cheat or attempting to bypass detection.

These constraints apply even if a prohibited approach would improve latency or
accuracy. Publisher permission is a separate policy question: being technically
read-only does not prove that simultaneous live use is officially permitted.

## Current risk posture

Offline analysis of recordings is preferred. Live desktop capture is
technically external and read-only, but no claim is made that the publisher or
anti-cheat cannot observe the existence of another running process.
