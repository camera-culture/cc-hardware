import socket
import threading

from hydra_config import HydraContainerConfig, config_wrapper
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import Registry
from cc_hardware.utils.manager import Component  # Assuming Component is defined in this module


@config_wrapper
class SocketComponentConfig(HydraContainerConfig, Registry):
    """
    Configuration for the SocketComponent.

    Attributes:
        host (str): Host/IP to bind the server socket. Defaults to 'localhost'.
        port (int): Port number to bind the server socket. Defaults to 9999.
        backlog (int): Maximum number of queued connections. Defaults to 1.
        buffer_size (int): Buffer size for receiving messages. Defaults to 1024 bytes.
    """
    host: str = "localhost"
    port: int = 9999
    backlog: int = 1
    buffer_size: int = 1024


class SocketComponent(Component[SocketComponentConfig]):
    """
    A TCP socket communicator component that waits for a 'START' message to become operational.
    Once started, the `is_okay` property returns True until a 'STOP' message is received.
    Communication is handled in a background thread.
    """

    def __init__(self, config: SocketComponentConfig):
        """
        Initializes the SocketComponent by setting up the server socket and starting the
        background thread to handle incoming messages.
        """
        super().__init__(config)
        self._state_lock = threading.Lock()
        self._running = False  # Flag indicating if 'START' has been received.
        self._stop_event = threading.Event()  # Signals the component to stop.
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.config.host, self.config.port))
        self._socket.listen(self.config.backlog)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """
        Runs in a background thread to handle TCP communication.

        Blocks on accepting a client connection and then listens for messages.
        Upon receiving 'START', the component becomes operational.
        Upon receiving 'STOP', the component stops.
        """
        try:
            conn, addr = self._socket.accept()
            get_logger().info(f"Accepted connection from {addr}.")
        except Exception:
            get_logger().exception("Failed to accept connection.")
            return

        with conn:
            while not self._stop_event.is_set():
                try:
                    data = conn.recv(self.config.buffer_size)
                    if not data:
                        # Client closed connection.
                        break
                    message = data.decode("utf-8").strip()
                    if message == "START":
                        with self._state_lock:
                            self._running = True
                        get_logger().info("Received 'START' message; component is now operational.")
                    elif message == "STOP":
                        with self._state_lock:
                            self._running = False
                        get_logger().info("Received 'STOP' message; stopping component.")
                        self._stop_event.set()
                        break
                except Exception:
                    get_logger().exception("Error during socket communication.")
                    break

    @property
    def is_okay(self) -> bool:
        """
        Checks if the component is operational.

        Returns:
            bool: True if 'START' has been received and 'STOP' has not been received; otherwise, False.
        """
        with self._state_lock:
            return self._running

    def close(self) -> None:
        """
        Closes the socket communicator and releases all resources.

        Signals the background thread to stop, closes the socket, and waits for the thread to finish.
        """
        self._stop_event.set()
        try:
            self._socket.close()
        except Exception:
            get_logger().exception("Error closing socket.")
        self._thread.join(timeout=2)
