"""Motion capture drivers for the cc-hardware package."""

from cc_hardware.drivers.mocap.mocap import (
    MotionCaptureSensor,
    MotionCaptureSensorConfig,
)

# =============================================================================
# Register the mocap sensor implementations

MotionCaptureSensor.register("ViveTrackerSensor", f"{__name__}.vive")
MotionCaptureSensorConfig.register("ViveTrackerSensorConfig", f"{__name__}.vive")
MotionCaptureSensorConfig.register(
    "ViveTrackerSensorConfig", f"{__name__}.vive", "ViveTrackerSensor"
)

MotionCaptureSensor.register("PklMotionCaptureSensor", f"{__name__}.pkl")
MotionCaptureSensorConfig.register("PklMotionCaptureSensorConfig", f"{__name__}.pkl")
MotionCaptureSensorConfig.register(
    "PklMotionCaptureSensorConfig", f"{__name__}.pkl", "PklMotionCaptureSensor"
)

# =============================================================================

__all__ = [
    "MotionCaptureSensor",
    "MotionCaptureSensorConfig",
]
