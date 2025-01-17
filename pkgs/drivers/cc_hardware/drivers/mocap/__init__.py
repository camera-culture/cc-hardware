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

# =============================================================================

__all__ = [
    "MotionCaptureSensor",
    "MotionCaptureSensorConfig",
]
