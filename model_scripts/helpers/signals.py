import signal

SHOULD_TERMINATE = False


def _handle_sigusr1(signum, frame):
    global SHOULD_TERMINATE
    print(f"[Signal] Received SIGUSR1 ({signum}). Requested graceful shutdown.")
    SHOULD_TERMINATE = True


def _handle_sigterm(signum, frame):
    global SHOULD_TERMINATE
    print(f"[Signal] Received SIGTERM ({signum}). Emergency shutdown requested.")
    SHOULD_TERMINATE = True


def install_signal_handlers():
    signal.signal(signal.SIGUSR1, _handle_sigusr1)
    signal.signal(signal.SIGTERM, _handle_sigterm)


def should_terminate() -> bool:
    return SHOULD_TERMINATE