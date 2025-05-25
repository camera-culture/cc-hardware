"""SPAD sensor drivers for the cc-hardware package."""

from cc_hardware.drivers.spads.spad import (
    SPADDataType,
    SPADSensor,
    SPADSensorConfig,
    SPADSensorData,
)

# =============================================================================
# Register the SPAD sensor implementations

SPADSensor.register("VL53L8CHSensor", f"{__name__}.vl53l8ch")
SPADSensorConfig.register("VL53L8CHConfig4x4", f"{__name__}.vl53l8ch")
SPADSensorConfig.register("VL53L8CHConfig8x8", f"{__name__}.vl53l8ch")
SPADSensorConfig.register("VL53L8CHConfig4x4", f"{__name__}.vl53l8ch", "VL53L8CHSensor")
SPADSensorConfig.register("VL53L8CHConfig8x8", f"{__name__}.vl53l8ch", "VL53L8CHSensor")

SPADSensor.register("TMF8828Sensor", f"{__name__}.tmf8828")
SPADSensorConfig.register("TMF8828Config", f"{__name__}.tmf8828")
SPADSensorConfig.register("TMF8828Config", f"{__name__}.tmf8828", "TMF8828Sensor")

SPADSensor.register("PklSPADSensor", f"{__name__}.pkl")
SPADSensorConfig.register("PklSPADSensorConfig", f"{__name__}.pkl")
SPADSensorConfig.register("PklSPADSensorConfig", f"{__name__}.pkl", "PklSPADSensor")

SPADSensor.register("SPADWrapper", f"{__name__}.spad_wrappers")
SPADSensorConfig.register("SPADWrapperConfig", f"{__name__}.spad_wrappers")
SPADSensorConfig.register(
    "SPADWrapperConfig", f"{__name__}.spad_wrappers", "SPADWrapper"
)
SPADSensor.register("SPADMergeWrapper", f"{__name__}.spad_wrappers")
SPADSensorConfig.register("SPADMergeWrapperConfig", f"{__name__}.spad_wrappers")
SPADSensorConfig.register(
    "SPADMergeWrapperConfig", f"{__name__}.spad_wrappers", "SPADMergeWrapper"
)
SPADSensor.register("SPADMovingAverageWrapper", f"{__name__}.spad_wrappers")
SPADSensorConfig.register("SPADMovingAverageWrapperConfig", f"{__name__}.spad_wrappers")
SPADSensorConfig.register(
    "SPADMovingAverageWrapperConfig",
    f"{__name__}.spad_wrappers",
    "SPADMovingAverageWrapper",
)

# =============================================================================

__all__ = [
    "SPADSensor",
    "SPADSensorConfig",
    "SPADDataType",
    "SPADSensorData",
]
