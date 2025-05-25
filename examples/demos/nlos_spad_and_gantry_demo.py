import threading

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import SnakeStepperController
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import Manager, register_cli, run_cli


@register_cli
def nlos_spad_and_gantry_demo(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    gantry: StepperMotorSystem,
):
    gantry_thread = None
    gantry_index = 0

    def setup(manager: Manager):
        _sensor = SPADSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

        _dashboard: SPADDashboard = dashboard.create_from_registry(
            config=dashboard, sensor=_sensor
        )
        _dashboard.setup()
        manager.add(dashboard=_dashboard)

        gantry_controller = SnakeStepperController(
            [
                dict(name="x", range=(0, 32), samples=2),
                dict(name="y", range=(0, 32), samples=2),
            ]
        )
        manager.add(gantry=gantry, controller=gantry_controller)

    def loop(
        frame: int,
        manager: Manager,
        sensor: SPADSensor,
        dashboard: SPADDashboard,
        gantry: StepperMotorSystem,
        controller: SnakeStepperController,
    ):
        nonlocal gantry_thread, gantry_index
        histograms = sensor.accumulate()
        dashboard.update(frame, histograms=histograms)

        if gantry_thread is None or not gantry_thread.is_alive():
            pos = controller.get_position(gantry_index % controller.total_positions)
            gantry_thread = threading.Thread(
                target=gantry.move_to, args=(pos["x"], pos["y"])
            )
            gantry_thread.start()
            gantry_index += 1

    def cleanup(gantry: StepperMotorSystem, **kwargs):
        if gantry_thread is not None and gantry_thread.is_alive():
            gantry_thread.join()
        gantry.move_to(0, 0)
        gantry.close()

    with Manager() as manager:
        try:
            manager.run(setup=setup, loop=loop, cleanup=cleanup)
        except KeyboardInterrupt:
            cleanup(manager.components["gantry"])


if __name__ == "__main__":
    run_cli(nlos_spad_and_gantry_demo)
