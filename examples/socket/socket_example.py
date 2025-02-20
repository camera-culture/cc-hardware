from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.socket_component import SocketComponentConfig, SocketComponent, SocketClientComponent, SocketClientComponentConfig


@register_cli
def socket_example(
    socket: SocketComponentConfig,
    client: bool,
    quiet: bool = False,
):
    def setup(manager: Manager):
        """Configures the manager with sensor instance.

        Args:
            manager (Manager): Manager to add sensor to.
        """
        if client:
            assert isinstance(socket, SocketClientComponentConfig), "Socket must be a client."
            _socket = SocketClientComponent.create_from_config(socket, name="SocketClientComponent")
            _socket.start()
            manager.add(socket=_socket)
        else:
            _socket = SocketComponent.create_from_config(socket, name="SocketComponent")
            get_logger().info(f"Waiting for socket on port {socket.port}...")
            _socket.wait()
            get_logger().info("Socket is ready.")
            manager.add(socket=_socket)

    def loop(
        frame: int,
        socket: SocketComponent | SocketClientComponent,
        **kwargs,
    ):
        """
        Args:
            frame (int): Current frame number.
            manager (Manager): Manager controlling the loop.
            sensor (MotionCaptureSensor): Sensor instance (unused here).
        """
        if not quiet:
            get_logger().info(f"Frame {frame}...")

        if client:
            socket.send_data("Hello, world!")
        else:
            data = socket.receive_data()
            get_logger().info(f"Received: {data}")


    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


def main():
    run_cli(socket_example)


if __name__ == "__main__":
    run_cli(socket_example)
