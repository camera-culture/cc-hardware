from cc_hardware.drivers import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli


@register_cli
def mocap_dashboard(
    sensor: MotionCaptureSensorConfig, dashboard: MotionCaptureDashboardConfig
):
    def setup(manager: Manager):
        """Configures the manager with sensor and dashboard instances.

        Args:
            manager (Manager): Manager to add sensor and dashboard to.
        """
        _sensor = MotionCaptureSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

        _dashboard = MotionCaptureDashboard.create_from_config(
            config=dashboard, sensor=_sensor
        )
        _dashboard.setup()
        manager.add(dashboard=_dashboard)

    def loop(
        frame: int,
        manager: Manager,
        sensor: MotionCaptureSensor,
        dashboard: MotionCaptureDashboard,
    ):
        """Updates dashboard each frame.

        Args:
            frame (int): Current frame number.
            manager (Manager): Manager controlling the loop.
            sensor (MotionCaptureSensor): Sensor instance (unused here).
            dashboard (MotionCaptureDashboard): Dashboard instance to update.
        """
        dashboard.update(frame)

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


def main():
    run_cli(mocap_dashboard)


if __name__ == "__main__":
    run_cli(mocap_viewer)
