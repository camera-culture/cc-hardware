from cc_hardware.drivers import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.utils import get_logger, register_cli, run_cli, Manager


@register_cli
def mocap_viewer(mocap: MotionCaptureSensorConfig):

    def setup(manager: Manager):
        _mocap = MotionCaptureSensor.create_from_config(mocap)
        manager.add(mocap=_mocap)

    def loop(iter: int, manager: Manager, mocap: MotionCaptureSensor) -> bool:
        get_logger().info(f"Frame {iter}...")

        pose = mocap.accumulate()
        print(pose)

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


if __name__ == "__main__":
    run_cli(mocap_viewer)
