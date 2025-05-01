import socket
import threading
import enum
import queue

from hydra_config import config_wrapper

from cc_hardware.utils import Config
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.manager import Component


class SocketComponentCommand(enum.IntEnum):
    """
    Enum to represent commands that can be sent to the SocketComponent.

    Attributes:
        START (int): Command to start the component.
        STOP (int): Command to stop the component.
        DATA (int): Command to send data to the component.
            The data command should then be followed by two additional messages.
            The first message should be the length of the data to be sent. It should
            be a 16 bit integer. The second message should be the data itself (of
            length specified in the first message).
    """

    START = 1
    STOP = 2
    DATA = 3


@config_wrapper
class SocketComponentConfig(Config):
    """
    Configuration for the SocketComponent.

    Attributes:
        host (str): Host/IP to bind the server socket. Defaults to 'localhost'.
        port (int): Port number to bind the server socket. Defaults to 9999.
        backlog (int): Maximum number of queued connections. Defaults to 1.
        buffer_size (int): Buffer size for receiving messages. Defaults to 1 byte.
        acknowledge (bool): Flag to acknowledge messages. Defaults to False.
    """

    host: str = "localhost"
    port: int = 9999
    backlog: int = 1
    buffer_size: int = 1
    ack: bool = False


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
        self._waiting = False
        self._data_queue: queue.Queue = queue.Queue()
        self._waiting_event = threading.Event()  # Signals the component to wait.
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
        get_logger().info(
            f"SocketComponent listening on {self.config.host}:{self.config.port}..."
        )

        while not self._stop_event.is_set():
            try:
                conn, addr = self._socket.accept()
                get_logger().info(f"Accepted connection from {addr}.")
            except Exception:
                get_logger().exception("Failed to accept connection.")
                return

            with conn:
                try:
                    data = conn.recv(self.config.buffer_size)
                    data = data.decode().strip()
                    if not data or not data.isdigit():
                        get_logger().warning(f"Received invalid message: '{data}'")
                        continue

                    cmd = SocketComponentCommand(int(data))
                    if cmd == SocketComponentCommand.START:
                        with self._state_lock:
                            self._running = True
                            if self._waiting:
                                self._waiting = False
                                self._waiting_event.set()
                        get_logger().info(
                            "Received 'START' message; component is now operational."
                        )
                    elif cmd == SocketComponentCommand.STOP:
                        with self._state_lock:
                            self._running = False
                            if self._waiting:
                                self._waiting = False
                                self._waiting_event.set()
                        if self.config.ack:
                            conn.send(b"ACK")
                        get_logger().info(
                            "Received 'STOP' message; stopping component."
                        )
                        self._stop_event.set()
                        break
                    elif cmd == SocketComponentCommand.DATA:
                        data_len = int.from_bytes(conn.recv(2), "big")
                        data = conn.recv(data_len)
                        get_logger().info(f"Received data: {data}")
                        self._data_queue.put(data)

                    if self.config.ack:
                        conn.send(b"ACK")
                except Exception:
                    get_logger().exception("Error during socket communication.")
                    break

        self._running = False
        self._stop_event.set()

    def wait(self) -> None:
        """
        Waits for the component to become operational.

        This method blocks until 'START' is received.
        """
        with self._state_lock:
            self._waiting = True
        get_logger().info("Waiting for 'START' message...")
        self._waiting_event.wait()
        self._waiting_event.clear()

    def receive_data(self) -> bytes:
        """
        Gets the next data message received.

        Returns:
            bytes: The data message received.
        """
        return self._data_queue.get()

    def has_data(self) -> bool:
        """
        Checks if there is data available in the queue.

        Returns:
            bool: True if there is data available; otherwise, False.
        """
        return not self._data_queue.empty()

    def start(self) -> None:
        with self._state_lock:
            self._running = True

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


@config_wrapper
class SocketClientComponentConfig(SocketComponentConfig):
    """
    Configuration for the SocketClientComponent.

    Inherits all attributes from SocketComponentConfig.

    Attributes:
        timeout (float): Timeout for socket operations in seconds. Defaults to 2.0.
    """

    timeout: float = 2.0


class SocketClientComponent(Component[SocketClientComponentConfig]):
    """
    A TCP socket client to communicate with the SocketComponent.

    Provides methods to send 'START', 'STOP', and 'DATA' commands.
    """

    def __init__(self, config: SocketClientComponentConfig):
        """
        Initializes the SocketClientComponent with the target host and port.
        """
        super().__init__(config)

        self._is_okay = True

    def _send_command(self, command: int, extra: bytes = b"") -> None:
        """
        Connects to the server, sends the command and any extra data, then closes the connection.

        Args:
            command (int): The command code to send.
            extra (bytes): Additional bytes to send after the command.
        """
        try:
            with socket.create_connection(
                (self.config.host, self.config.port), timeout=self.config.timeout
            ) as sock:
                # Send command code as a string.
                sock.sendall(str(command).encode())
                if extra:
                    sock.sendall(extra)

                if self.config.ack:
                    ack = sock.recv(3)
                    if ack != b"ACK":
                        get_logger().warning(f"Received invalid acknowledgment: '{ack}'")
        except Exception:
            get_logger().exception("Error during socket communication.")
            self._is_okay = False

    def start(self) -> None:
        """
        Sends a 'START' command to the server to activate the component.
        """
        get_logger().info("Sending 'START' command.")
        self._send_command(SocketComponentCommand.START)

    def stop(self) -> None:
        """
        Sends a 'STOP' command to the server to deactivate the component.
        """
        get_logger().info("Sending 'STOP' command.")
        self._send_command(SocketComponentCommand.STOP)

    def send_data(self, data: str | bytes) -> None:
        """
        Sends a 'DATA' command followed by the length and the actual data.

        The protocol:
            1. Send the 'DATA' command.
            2. Send a 2-byte zero-padded ASCII representation of the length of data.
            3. Send the data.

        Args:
            data (bytes): The data to send.
        """
        if isinstance(data, str):
            data = data.encode()

        data_length = len(data)
        if data_length > 99:
            get_logger().error("Data length exceeds 2-digit limit.")
            return
        # Format length as a 2-digit zero-padded string.
        length_str = f"{data_length:02d}".encode()
        get_logger().info(
            f"Sending 'DATA' command with payload of length {data_length}."
        )
        self._send_command(SocketComponentCommand.DATA, extra=length_str + data)

    @property
    def is_okay(self) -> bool:
        """
        Checks if the component is operational.
        """
        return self._is_okay

    def close(self) -> bool:
        """
        Checks if the client is closed.

        Noop for the client.
        """
        return False


SocketComponentConfig.register("SocketComponentConfig", __name__)
SocketComponentConfig.register("SocketComponentConfig", __name__, "SocketComponent")
SocketClientComponentConfig.register("SocketClientComponentConfig", __name__)
SocketClientComponentConfig.register("SocketClientComponentConfig", __name__, "SocketClientComponent")
