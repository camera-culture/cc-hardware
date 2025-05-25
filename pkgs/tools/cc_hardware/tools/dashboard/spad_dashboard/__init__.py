"""Dashboards for SPAD sensors."""

from cc_hardware.tools.dashboard.spad_dashboard.spad_dashboard import (
    SPADDashboard,
    SPADDashboardConfig,
)

# =============================================================================
# Register the dashboard implementations

SPADDashboard.register("PyQtGraphDashboard", f"{__name__}.pyqtgraph")
SPADDashboardConfig.register("PyQtGraphDashboardConfig", f"{__name__}.pyqtgraph")
SPADDashboardConfig.register(
    "PyQtGraphDashboardConfig", f"{__name__}.pyqtgraph", "PyQtGraphDashboard"
)
SPADDashboardConfig.register("PyQtGraphDashboardConfig", f"{__name__}.pyqtgraph")
SPADDashboardConfig.register(
    "PyQtGraphDashboardConfig", f"{__name__}.pyqtgraph", "PyQtGraphDashboard"
)

# =============================================================================

__all__ = [
    "SPADDashboard",
    "SPADDashboardConfig",
]
