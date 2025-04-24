import os
# disable all Hydra file logging
os.environ["HYDRA_HYDRA_LOGGING__FILE"] = "false"
os.environ["HYDRA_JOB_LOGGING__FILE"] = "false"

import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import SnakeStepperController
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.manager import Manager

NOW = datetime.now()


def setup(
    manager: Manager,
    *,
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    gantry: StepperMotorSystem,
    x_samples: int,
    y_samples: int,
    logdir: Path,
    object: str,
    spad_position: Tuple[float, float, float],
):
    logdir.mkdir(parents=True, exist_ok=True)

    spad = SPADSensor.create_from_config(sensor)
    if not spad.is_okay:
        get_logger().fatal("Failed to initialize spad")
        return
    manager.add(spad=spad)

    dashboard = SPADDashboard.create_from_config(dashboard, sensor=spad)
    dashboard.setup()
    manager.add(dashboard=dashboard)

    gantry_controller = SnakeStepperController(
        [
            dict(name="x", range=(0, 32), samples=x_samples),
            dict(name="y", range=(0, 32), samples=y_samples),
        ]
    )
    manager.add(gantry=gantry, controller=gantry_controller)

    output_pkl = logdir / f"{object}_data.pkl"
    assert not output_pkl.exists(), f"Output file {output_pkl} already exists"
    pkl_writer = PklHandler(output_pkl)
    manager.add(writer=pkl_writer)

    # Write top-level metadata once
    pkl_writer.append({
        "metadata": {
            "object": object,
            "spad_position": {
                "x": spad_position[0],
                "y": spad_position[1],
                "z": spad_position[2],
            },
            "start_time": NOW.isoformat(),
        }
    })


def loop(
    iter: int,
    manager: Manager,
    spad: SPADSensor,
    dashboard: SPADDashboard,
    controller: SnakeStepperController,
    gantry: StepperMotorSystem,
    writer: PklHandler,
    **kwargs,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    histogram = spad.accumulate()
    dashboard.update(iter, histograms=histogram)

    pos = controller.get_position(iter)
    if pos is None:
        return False

    gantry.move_to(pos["x"], pos["y"])

    writer.append(
        {
            "iter": iter,
            "pos": pos,
            "histogram": histogram,
        }
    )

    time.sleep(0.25)
    return True


def cleanup(gantry: StepperMotorSystem, **kwargs):
    get_logger().info("Cleaning up...")
    gantry.move_to(0, 0)
    gantry.close()


@register_cli
def spad_gantry_capture_v2(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    gantry: StepperMotorSystem,
    x_samples: int,
    y_samples: int,
    object: str,
    spad_position: Tuple[float, float, float],
    logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S"),
):
    _setup = partial(
        setup,
        sensor=sensor,
        dashboard=dashboard,
        gantry=gantry,
        logdir=logdir,
        x_samples=x_samples,
        y_samples=y_samples,
        object=object,
        spad_position=spad_position,
    )

    with Manager() as manager:
        try:
            manager.run(setup=_setup, loop=loop, cleanup=cleanup)
        except KeyboardInterrupt:
            cleanup(manager.components["gantry"])
        finally:
            # Big, bold, green printout of the final .pkl path
            print(
                f"\033[1;32mPKL file saved to "
                f"{(logdir / f'{object}_data.pkl').resolve()}\033[0m"
            )


if __name__ == "__main__":
    run_cli(spad_gantry_capture_v2)
