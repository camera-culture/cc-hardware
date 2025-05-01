import time
from pathlib import Path
from datetime import datetime

from cc_hardware.drivers import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.socket_component import SocketComponentConfig, SocketComponent

NOW = datetime.now()


@register_cli
def mocap_server(
    sensor: MotionCaptureSensorConfig,
    socket: SocketComponentConfig | None = None,
    logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d"),
    force: bool = False,
    quiet: bool = False,
):
    writer: PklHandler | None = None 

    def setup(manager: Manager):
        """Attach sensor & optional socket; no blocking waits."""
        _sensor = MotionCaptureSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

        if socket is not None:
            _socket = SocketComponent.create_from_config(socket, name="SocketComponent")
            _socket.start()
            manager.add(socket=_socket)

    def _process_data(data: str):
        _stop()
        if data == "stop":
            return

        _start(Path(data))

    def _start(filename: Path):
        get_logger().info("Starting recording.")

        nonlocal writer
        if writer is not None:  # close previous session
            writer.close()
        _pkl = logdir / filename
        assert _pkl.suffix == ".pkl", "Filename must have .pkl extension."
        assert force or not _pkl.exists(), "File already exists."
        logdir.mkdir(exist_ok=True, parents=True)
        writer = PklHandler(_pkl)
        get_logger().info(f"Recording to {_pkl}")

    def _stop():
        get_logger().info("Stopping recording.")

        nonlocal writer
        if writer is not None:
            writer.close()
            writer = None

    def loop(
        frame: int,
        manager: Manager,
        sensor: MotionCaptureSensor,
        socket: SocketComponent | None = None,
        **kwargs,
    ):
        nonlocal writer

        # start a new session when socket supplies a filename
        if socket is not None and socket.has_data():
            _process_data(socket.receive_data().decode().strip())
            manager.iter = 0

        # nothing to do if not currently recording
        if writer is None:
            return

        if not quiet:
            get_logger().info(f"Frame {frame}...")

        data = sensor.accumulate()
        if data is None:
            get_logger().warning("Bad data.")
            return

        writer.append({"frame": frame, "data": data})

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


def main():
    run_cli(mocap_server)


if __name__ == "__main__":
    run_cli(mocap_server)
