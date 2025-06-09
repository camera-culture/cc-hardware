import time
from functools import partial
from pathlib import Path

import torch
from gui.dashboard import CVPR25Dashboard, CVPR25DashboardConfig
from ml.model import DeepLocation8

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import (
    SnakeControllerAxisConfig,
    SnakeStepperControllerConfigXY,
    StepperController,
)
from cc_hardware.drivers.stepper_motors.telemetrix_stepper import (
    SingleDrive1AxisGantryConfig,
)
from cc_hardware.utils import (
    AtomicVariable,
    Manager,
    ThreadedComponent,
    get_logger,
    register_cli,
    run_cli,
    threaded_component,
)
from cc_hardware.utils.constants import TORCH_DEVICE
from cc_hardware.utils.file_handlers import PklReader

# ==========

STEPPER_SYSTEM = SingleDrive1AxisGantryConfig.create()
STEPPER_CONTROLLER = SnakeStepperControllerConfigXY.create(
    axes=dict(
        x=SnakeControllerAxisConfig(range=(0, 32), samples=3),
        y=SnakeControllerAxisConfig(range=(0, 32), samples=2),
    )
)

GUI = CVPR25DashboardConfig.create(
    x_range=(0, 32),
    y_range=(0, 32),
    point_size=10.0,
)

# ==========

CONTROLLER_POS = AtomicVariable(dict(x=0, y=0))
HISTOGRAM = AtomicVariable(None)
HISTOGRAMS = AtomicVariable([])

# ==========


def stepper_callback(
    future,
    *,
    manager: Manager,
    stepper_system: StepperMotorSystem,
    controller: StepperController,
    i: int,
    repeat: bool = True,
):
    if not manager.is_looping:
        get_logger().info("Manager is not looping, stopping stepper callback.")
        return

    if repeat:
        i %= controller.total_positions.result()

    pos = controller.get_position(i).result()
    CONTROLLER_POS.set(pos)
    stepper_system.move_to(pos["x"], pos["y"]).add_done_callback(
        partial(
            stepper_callback,
            manager=manager,
            stepper_system=stepper_system,
            controller=controller,
            i=i + 1,
        )
    )


def sensor_callback(
    future,
    *,
    manager: Manager,
    sensor: SPADSensor,
):
    if not manager.is_looping:
        get_logger().info("Manager is not looping, stopping stepper callback.")
        return

    data = future.result()
    assert SPADDataType.HISTOGRAM in data, "Sensor must support histogram data type."
    HISTOGRAM.set(data[SPADDataType.HISTOGRAM])

    histograms = HISTOGRAMS.get()
    histograms.append(data[SPADDataType.HISTOGRAM])
    if len(histograms) > 10:
        histograms.pop(0)
    HISTOGRAMS.set(histograms)

    sensor.accumulate().add_done_callback(
        partial(
            sensor_callback,
            manager=manager,
            sensor=sensor,
        )
    )


# ==========


def setup(
    manager: Manager,
    config_path: Path,
    model_path: Path,
    sensor_port: str | None = None,
    stepper_port: str | None = None,
    background: bool = False,
):
    """Configures the manager with sensor and dashboard instances.

    Args:
        manager (Manager): Manager to add sensor and dashboard to.
    """
    config = PklReader.load_all(config_path)
    assert len(config) == 1, "Expected exactly one configuration in the pickle file."
    config = config[0]

    assert "sensor" in config, "Configuration must contain 'sensor' key."
    sensor: SPADSensorConfig = config["sensor"]
    sensor.window_size = 10
    _sensor = threaded_component(
        SPADSensor.create_from_config(sensor, port=sensor_port)
    )
    manager.add(sensor=_sensor)

    _sensor.accumulate().add_done_callback(
        partial(
            sensor_callback,
            manager=manager,
            sensor=_sensor,
        )
    )

    _controller = threaded_component(
        StepperController.create_from_config(STEPPER_CONTROLLER)
    )
    manager.add(controller=_controller)

    if stepper_port is not None:
        STEPPER_SYSTEM.port = stepper_port
    _stepper_system = threaded_component(
        StepperMotorSystem.create_from_config(STEPPER_SYSTEM)
    )
    _stepper_system.initialize().result()
    manager.add(stepper_system=_stepper_system)

    _stepper_system.move_to(0, 0).add_done_callback(
        partial(
            stepper_callback,
            manager=manager,
            stepper_system=_stepper_system,
            controller=_controller,
            i=0,
        )
    )

    _gui = CVPR25Dashboard(GUI)
    _gui.setup()
    manager.add(gui=_gui)

    assert model_path.exists(), f"Model file {model_path} does not exist."
    model = DeepLocation8(sensor.height, sensor.width, sensor.num_bins).to(TORCH_DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(TORCH_DEVICE)
    manager.add(model=model)


def loop(
    frame: int,
    manager: Manager,
    sensor: SPADSensor | ThreadedComponent,
    stepper_system: StepperMotorSystem | ThreadedComponent,
    model: DeepLocation8,
    gui: CVPR25Dashboard,
    **kwargs,
):
    histogram = HISTOGRAM.get()
    if len(histogram) == 0:
        get_logger().warning("No histogram available for evaluation.")
        return
    positions = model.evaluate(histogram)

    gui.update(
        frame=frame,
        positions=positions,
    )


def cleanup(
    stepper_system: StepperMotorSystem,
    **kwargs,
):
    get_logger().info("Cleaning up...")
    stepper_system.move_to(0, 0)
    stepper_system.close()


@register_cli
def cvpr2025(
    config: Path,
    model: Path,
    sensor_port: str | None = None,
    stepper_port: str | None = None,
    background: bool = False,
):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    global t0
    t0 = time.time()

    with Manager() as manager:
        manager.run(
            setup=partial(
                setup,
                config_path=config,
                model_path=model,
                sensor_port=sensor_port,
                stepper_port=stepper_port,
                background=background,
            ),
            loop=loop,
            cleanup=cleanup,
        )


if __name__ == "__main__":
    run_cli(cvpr2025)
