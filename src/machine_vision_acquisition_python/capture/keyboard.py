import typing
import threading
import atexit
from sshkeyboard import listen_keyboard, stop_listening

HANDLERS: typing.Dict[str, typing.Callable] = {}


def press(key):
    global HANDLERS
    handler = HANDLERS.get(key)
    if handler:
        handler()


def register_callback(key: str, callback: typing.Callable):
    """Register a callback for key press events. Use functools.partial for arguments"""
    global HANDLERS
    if len(key) != 1:
        raise ValueError("must register on single character")
    key = key.lower()
    if HANDLERS.get(key) is not None:
        raise ValueError(f"{key} already has a registerd callback")
    HANDLERS.update({key: callback})


keyboard_thread = threading.Thread(
    target=listen_keyboard,
    args=(),
    kwargs={
        "on_press": press,
        # "on_release": release,
        "lower": True,
    },
    daemon=True,
)

# ensure thread is shutdown
atexit.register(stop_listening)
# start keyboard thread
keyboard_thread.start()
