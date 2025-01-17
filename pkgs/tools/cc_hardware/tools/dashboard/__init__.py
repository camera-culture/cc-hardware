from cc_hardware.tools.dashboard.dashboard import Dashboard, DashboardConfig
from cc_hardware.tools.dashboard.spad_dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import Manager, register_cli, run_cli

# =============================================================================


@register_cli
def dashboard(dashboard: DashboardConfig):
    def setup(manager: Manager):
        """Configures the manager with sensor and dashboard instances.

        Args:
            manager (Manager): Manager to add sensor and dashboard to.
        """
        _dashboard = Dashboard.create_from_config(dashboard)
        _dashboard.setup()
        manager.add(dashboard=_dashboard)

    def loop(frame: int, manager: Manager, dashboard: Dashboard):
        """Updates dashboard each frame.

        Args:
            frame (int): Current frame number.
            manager (Manager): Manager controlling the loop.
            dashboard (SPADDashboard): Dashboard instance to update.
        """
        dashboard.update(frame)

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)

def main():
    run_cli(dashboard)

# =============================================================================

__all__ = [
    "Dashboard",
    "DashboardConfig",
    # spad_dashboard
    "SPADDashboard",
    "SPADDashboardConfig",
]
