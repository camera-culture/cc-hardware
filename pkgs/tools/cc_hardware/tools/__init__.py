"""Command line interface for the cc_hardware.tools package."""

from cc_hardware.utils import Registry

# =============================================================================


class ToolRegistry(Registry):
    pass


ToolRegistry.register("dashboard", f"{__name__}.dashboard")
ToolRegistry.register("spad_dashboard", f"{__name__}.dashboard.spad_dashboard")
ToolRegistry.register("mocap_dashboard", f"{__name__}.dashboard.mocap_dashboard")
ToolRegistry.register("jogger", f"{__name__}.jogger")
ToolRegistry.register("camera_viewer", f"{__name__}.camera_viewer")
ToolRegistry.register("tmf8828_flash", f"{__name__}.flash")
ToolRegistry.register("vl53l8ch_flash", f"{__name__}.flash")
ToolRegistry.register("tmf8828_calibrate", f"{__name__}.calibration")

# =============================================================================


def main():
    import argparse
    import sys

    from cc_hardware.utils import get_logger, run_cli

    parser = argparse.ArgumentParser(description="cc_hardware tools", add_help=False)

    parser.add_argument(
        "cmd",
        help="The command to run",
        choices=list(ToolRegistry.registry.keys()),
    )

    known_args, unknown_args = parser.parse_known_args()

    # Remove known args from argv
    sys.argv = sys.argv[:1] + unknown_args

    get_logger().info(f"Running command: {known_args.cmd}")
    run_cli(ToolRegistry.registry[known_args.cmd])
