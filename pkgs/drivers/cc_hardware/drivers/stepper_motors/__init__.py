"""Stepper motor drivers for the cc-hardware package."""

from cc_hardware.drivers.stepper_motors.stepper_motor import (
    DummyStepperMotor,
    StepperMotor,
    StepperMotorConfig
)
from cc_hardware.drivers.stepper_motors.stepper_system import (
    StepperMotorSystem,
    StepperMotorSystemConfig,
    StepperMotorSystemAxis,
)

# Register the stepper motor implementations
# StepperMotorSystemConfig.register("KinesisStepperMotorSystem", f"{__name__}.kinesis_stepper")


StepperMotorSystem.register("TelemetrixStepperMotorSystem", f"{__name__}.telemetrix_stepper")
StepperMotorSystemConfig.register("TelemetrixStepperMotorSystemConfig", f"{__name__}.telemetrix_stepper")
StepperMotorSystemConfig.register("TelemetrixStepperMotorSystemConfig", f"{__name__}.telemetrix_stepper", "TelemetrixStepperMotorSystem")

StepperMotorSystem.register("SingleDrive1AxisGantryConfig", f"{__name__}.telemetrix_stepper", "TelemetrixStepperMotorSystem")
StepperMotorSystemConfig.register("SingleDrive1AxisGantryConfig", f"{__name__}.telemetrix_stepper")
StepperMotorSystemConfig.register("SingleDrive1AxisGantryConfig", f"{__name__}.telemetrix_stepper", "TelemetrixStepperMotorSystem")
StepperMotorSystemConfig.register("SingleDrive1AxisGantryXConfig", f"{__name__}.telemetrix_stepper")
StepperMotorSystemConfig.register("SingleDrive1AxisGantryXConfig", f"{__name__}.telemetrix_stepper", "TelemetrixStepperMotor")
StepperMotorSystemConfig.register("SingleDrive1AxisGantryYConfig", f"{__name__}.telemetrix_stepper")
StepperMotorSystemConfig.register("SingleDrive1AxisGantryYConfig", f"{__name__}.telemetrix_stepper", "TelemetrixStepperMotor")

StepperMotorSystemConfig.register("DualDrive2AxisGantryConfig", f"{__name__}.telemetrix_stepper")

__all__ = [
    "StepperMotor",
    "StepperMotorConfig",
    "DummyStepperMotor",
    "StepperMotorSystem",
    "StepperMotorSystemConfig",
    "StepperMotorSystemAxis",
]
