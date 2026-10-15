"""Low-latency Windows screen capture for one visible window."""
from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import threading
import time

import numpy as np

from .capture import Frame


# DXcam 0.3.0 has an upstream Windows/comtypes defect in DXCamera.release()
# (ra1nty/DXcam#144). Keep the single stopped camera alive until interpreter
# shutdown; comtypes then tears COM down in the safe order. It can also be
# restarted if this module opens capture again in the same process.
_STOPPED_DXCAM_CAMERAS = []


def _window_handle(title):
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if hwnd:
        return hwnd
    matches = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def callback(candidate, _):
        length = user32.GetWindowTextLengthW(candidate)
        if length and user32.IsWindowVisible(candidate):
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(candidate, buffer, length + 1)
            if title.casefold() in buffer.value.casefold():
                matches.append(candidate)
        return True

    user32.EnumWindows(callback_type(callback), 0)
    if len(matches) != 1:
        raise RuntimeError(f'Expected one visible window containing {title!r}, found {len(matches)}')
    return matches[0]


def window_client_region(title):
    """Return client rectangle in physical virtual-desktop coordinates."""
    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        user32.SetProcessDPIAware()
    hwnd = _window_handle(title)
    if user32.IsIconic(hwnd):
        raise RuntimeError(f'Window {title!r} is minimized')
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError()
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise ctypes.WinError()
    region = (origin.x, origin.y, origin.x + rect.right, origin.y + rect.bottom)
    if region[2] <= region[0] or region[3] <= region[1]:
        raise RuntimeError(f'Window {title!r} has an empty client area')
    return hwnd, region


def intersection_area(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


class LatestScreenCapture:
    def __init__(self, window_title='Champions', fps=60):
        self.window_title, self.fps = window_title, int(fps)
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.latest = None
        self.error = None
        self.done = False
        self.info = {}
        self.thread = threading.Thread(target=self._run, daemon=True, name='screen-capture')

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        camera = None
        try:
            import dxcam
            _, client = window_client_region(self.window_title)
            choices = []
            # Inspect DXcam's already-enumerated outputs first. Creating a camera for
            # every monitor allocates multiple Desktop Duplication/COM objects and can
            # produce noisy comtypes cleanup errors at process shutdown.
            factory = dxcam.__dict__.get('__factory')
            if factory is None or not factory.outputs:
                raise RuntimeError('DXcam did not enumerate any display outputs')
            for output_index, output in enumerate(factory.outputs[0]):
                output.update_desc()
                desktop = output.desc.DesktopCoordinates
                monitor = (desktop.left, desktop.top, desktop.right, desktop.bottom)
                area = intersection_area(client, monitor)
                choices.append((area, output_index, monitor))
            if not choices or max(item[0] for item in choices) == 0:
                raise RuntimeError(f'Window {self.window_title!r} is outside capturable outputs')
            _, output_index, monitor = max(choices, key=lambda item: item[0])
            camera = dxcam.create(output_idx=output_index, output_color='BGR',
                                  max_buffer_len=2, backend='dxgi')
            clipped = (max(client[0], monitor[0]), max(client[1], monitor[1]),
                       min(client[2], monitor[2]), min(client[3], monitor[3]))
            region = (clipped[0] - monitor[0], clipped[1] - monitor[1],
                      clipped[2] - monitor[0], clipped[3] - monitor[1])
            camera.start(region=region, target_fps=self.fps, video_mode=False)
            self.info = {'backend': 'DXGI Desktop Duplication', 'window': self.window_title,
                         'output_index': output_index, 'monitor_region': list(monitor),
                         'window_client_region': list(client), 'capture_region': list(region),
                         'requested_fps': self.fps,
                         'timestamp_origin': 'after DXcam delivery; display scanout latency unknown'}
            seq = 0
            while not self.stop_event.is_set():
                item = camera.get_latest_frame(with_timestamp=True)
                if item is None:
                    continue
                image, source_time = item
                received_at = time.perf_counter()
                # DXcam owns its ring buffer; keep a caller-owned frame beyond this iteration.
                frame = Frame(seq, received_at, np.ascontiguousarray(image), float(source_time))
                self.info['actual_size'] = [image.shape[1], image.shape[0]]
                with self.condition:
                    self.latest = frame
                    self.condition.notify_all()
                seq += 1
        except Exception as exc:
            self.error = exc
        finally:
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass
                if camera not in _STOPPED_DXCAM_CAMERAS:
                    _STOPPED_DXCAM_CAMERAS.append(camera)
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def get(self, after=-1, timeout=1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.error or self.done or (self.latest is not None and self.latest.seq > after),
                timeout,
            )
            if self.error:
                raise RuntimeError('Screen capture failed') from self.error
            if self.latest is not None and self.latest.seq > after:
                return self.latest
            return None

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            print('Warning: screen capture thread is still stopping.')
