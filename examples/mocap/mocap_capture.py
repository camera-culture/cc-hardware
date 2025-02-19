from pathlib import Path
from datetime import datetime

from cc_hardware.drivers import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler

NOW = datetime.now()


@register_cli
def mocap_capture(
    sensor: MotionCaptureSensorConfig,
    pkl: Path | None = None,
    logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d"),
    force: bool = False,
):
    def setup(manager: Manager):
        """Configures the manager with sensor instance.

        Args:
            manager (Manager): Manager to add sensor to.
        """
        if pkl is not None:
            _pkl = logdir / pkl
            assert _pkl.suffix == ".pkl", "Filename must have .pkl extension."
            assert force or not _pkl.exists(), "File already exists."
            logdir.mkdir(exist_ok=True, parents=True)
            manager.add(writer=PklHandler(_pkl))

        _sensor = MotionCaptureSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

    def loop(
        frame: int,
        manager: Manager,
        sensor: MotionCaptureSensor,
        writer: PklHandler | None = None,
    ):
        """
        Args:
            frame (int): Current frame number.
            manager (Manager): Manager controlling the loop.
            sensor (MotionCaptureSensor): Sensor instance (unused here).
        """
        get_logger().info(f"Frame {frame}...")

        data = sensor.accumulate()
        if data is None:
            get_logger().warning("Got bad data. Stopping...")
            return False

        if writer is not None:
            get_logger().info("\tWriting...")
            writer.append(
                {
                    "frame": frame,
                    "data": data,
                }
            )

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


def main():
    run_cli(mocap_capture)


if __name__ == "__main__":
    run_cli(mocap_capture)
