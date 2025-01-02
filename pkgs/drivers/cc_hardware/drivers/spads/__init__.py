"""SPAD sensor drivers for the cc-hardware package."""

from cc_hardware.drivers.spads.dashboards import SPADDashboard, SPADDashboardConfig
from cc_hardware.drivers.spads.spad import SPADSensor, SPADSensorConfig

# =============================================================================
# Register the SPAD sensor implementations

SPADSensor.register("VL53L8CHSensor", f"{__name__}.vl53l8ch")
SPADSensorConfig.register("VL53L8CHConfig", f"{__name__}.vl53l8ch")
SPADSensorConfig.register("VL53L8CHConfig4x4", f"{__name__}.vl53l8ch")
SPADSensorConfig.register("VL53L8CHConfig8x8", f"{__name__}.vl53l8ch")

SPADSensor.register("TMF8828Sensor", f"{__name__}.tmf8828")
SPADSensorConfig.register("TMF8828Config", f"{__name__}.tmf8828")

SPADSensor.register("PklSPADSensor", f"{__name__}.pkl")
SPADSensorConfig.register("PklSPADConfig", f"{__name__}.pkl")

# =============================================================================

__all__ = [
    "SPADSensor",
    "SPADSensorConfig",
    "SPADDashboard",
    "SPADDashboardConfig",
]
