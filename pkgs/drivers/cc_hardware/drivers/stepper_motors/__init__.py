"""Stepper motor drivers for the cc-hardware package."""

from cc_hardware.drivers.stepper_motors.stepper_motor import (
    DummyStepperMotor,
    StepperMotor,
)
from cc_hardware.drivers.stepper_motors.stepper_system import (
    StepperMotorSystem,
    StepperMotorSystemAxis,
)

# Register the stepper motor implementations
StepperMotorSystem.register("KinesisStepperMotorSystem", f"{__name__}.kinesis_stepper")
StepperMotorSystem.register(
    "TelemetrixStepperMotorSystem", f"{__name__}.telemetrix_stepper"
)
StepperMotorSystem.register("SingleDrive1AxisGantry", f"{__name__}.telemetrix_stepper")
StepperMotorSystem.register("DualDrive2AxisGantry", f"{__name__}.telemetrix_stepper")

__all__ = [
    "StepperMotor",
    "DummyStepperMotor",
    "StepperMotorSystem",
    "StepperMotorSystemAxis",
]
