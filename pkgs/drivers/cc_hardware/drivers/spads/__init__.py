"""SPAD sensor drivers for the cc-hardware package."""

from cc_hardware.drivers.spads.spad import SPADSensor, SPADSensorConfig

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

# =============================================================================

__all__ = [
    "SPADSensor",
    "SPADSensorConfig",
]
