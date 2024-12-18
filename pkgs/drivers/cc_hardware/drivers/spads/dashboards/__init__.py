from cc_hardware.drivers.spads.dashboards.dashboard import SPADDashboard

# Register the dashboard implementations
SPADDashboard.register("PyQtGraphDashboard", f"{__name__}.pyqtgraph")
SPADDashboard.register("MatplotlibDashboard", f"{__name__}.matplotlib")
SPADDashboard.register("DashDashboard", f"{__name__}.dash")

__all__ = [
    "SPADDashboard",
]
