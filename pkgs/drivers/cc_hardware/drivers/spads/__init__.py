"""SPAD sensor drivers for the cc-hardware package."""

from cc_hardware.drivers.spads.dashboards import SPADDashboard
from cc_hardware.drivers.spads.spad import SPADSensor

# Register the SPAD sensor implementations
SPADSensor.register("VL53L8CHSensor", f"{__name__}.vl53l8ch")
SPADSensor.register("TMF8828Sensor", f"{__name__}.tmf8828")
SPADSensor.register("PklSPADSensor", f"{__name__}.pkl")

__all__ = [
    "SPADSensor",
    "SPADDashboard",
]
