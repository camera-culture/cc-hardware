from cc_hardware.drivers.spads import SPADSensorConfig
from cc_hardware.tools.cli import register_cli, run_cli
from cc_hardware.utils.manager import Manager
from cc_hardware.utils.plotting import transient_gui


@register_cli
def transient_viewer(
    sensor: SPADSensorConfig,
    show: bool = True,
    save: bool = False,
    filename: str | None = None,
    min_bin: int | None = None,
    max_bin: int | None = None,
    fullscreen: bool = False,
    fps: int = 10,
    normalize_per_pixel: bool = True,
):
    """View transient data from a sensor. Renders as a video."""

    def setup(manager: Manager):
        nonlocal sensor
        sensor = sensor.create_instance()

        transient_gui(
            sensor,
            show=show,
            save=save,
            filename=filename,
            fps=fps,
            fullscreen=fullscreen,
            min_bin=min_bin,
            max_bin=max_bin,
            normalize_per_pixel=normalize_per_pixel,
        )

    with Manager() as manager:
        manager.run(setup=setup)


if __name__ == "__main__":
    run_cli(transient_viewer)
