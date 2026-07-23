import socket
from typing import Tuple


def is_port_available(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True

def reserve_open_port(
    start: int = 9000,
    end: int = 9500,
    host: str = "0.0.0.0",
) -> Tuple[socket.socket, int]:
    """
    Atomically reserves a free port in the given range.
    Returns (socket, port).
    The socket MUST be kept open until the server binds.
    """
    for port in range(start, end + 1):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            s.bind((host, port))
            s.listen(1)  # reserve
            return s, port
        except OSError:
            s.close()

    raise RuntimeError(
        f"No free ports available in range {start}-{end} on {host}"
    )

def is_port_open(port: int, host="localhost") -> bool:
    """Check if a TCP port is open"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        try:
            sock.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout):
            return False