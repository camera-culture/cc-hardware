from cc_hardware.drivers.spads.dashboards.dashboard import (
    SPADDashboard,
    SPADDashboardConfig,
)

# =============================================================================
# Register the dashboard implementations

SPADDashboard.register("PyQtGraphDashboard", f"{__name__}.pyqtgraph")
SPADDashboardConfig.register("PyQtGraphDashboardConfig", f"{__name__}.pyqtgraph")

SPADDashboard.register("MatplotlibDashboard", f"{__name__}.matplotlib")
SPADDashboardConfig.register("MatplotlibDashboardConfig", f"{__name__}.matplotlib")

SPADDashboard.register("DashDashboard", f"{__name__}.dash")
SPADDashboardConfig.register("DashDashboardConfig", f"{__name__}.dash")

# =============================================================================

__all__ = [
    "SPADDashboard",
    "SPADDashboardConfig",
]
