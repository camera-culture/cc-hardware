"""Dashboards for SPAD sensors."""

from cc_hardware.tools.dashboard.spad_dashboard.spad_dashboard import (
    DummySPADDashboard,
    DummySPADDashboardConfig,
    SPADDashboard,
    SPADDashboardConfig,
)

# =============================================================================
# Register the dashboard implementations

SPADDashboard.register("DummySPADDashboard", f"{__name__}.spad_dashboard")
SPADDashboardConfig.register("DummySPADDashboardConfig", f"{__name__}.spad_dashboard")
SPADDashboardConfig.register(
    "DummySPADDashboardConfig", f"{__name__}.spad_dashboard", "DummySPADDashboard"
)

SPADDashboard.register("PyQtGraphDashboard", f"{__name__}.pyqtgraph")
SPADDashboardConfig.register("PyQtGraphDashboardConfig", f"{__name__}.pyqtgraph")
SPADDashboardConfig.register(
    "PyQtGraphDashboardConfig", f"{__name__}.pyqtgraph", "PyQtGraphDashboard"
)
# =============================================================================

__all__ = [
    "SPADDashboard",
    "SPADDashboardConfig",
    "DummySPADDashboard",
    "DummySPADDashboardConfig",
]
